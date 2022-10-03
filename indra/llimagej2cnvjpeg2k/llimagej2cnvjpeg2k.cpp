/*
	HIGHLY EXPERIMENTAL	nvJPEG2000 decoder
	
	Requires install of cuda, nvcc and libnvjpeg2k >=v0.5
	Currently works on Linux
	As of nvJPEG2000 v0.5.0, any baked textures and some regular textures do not work

	Add to build script: export CUDACXX=nvcc
	Pass "-DUSE_NVJPEG2K:BOOL=On" to cmake/autobuild
	
	This was a quick implementation to see if this decoder would work at all and
	the implementation is non-optimal, the cuda streams are reusable and there
	is some overhead on remaking them for each texture.
	
*/

#include "linden_common.h"
#include "llimagej2cnvjpeg2k.h"

int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
int dev_free(void *p) { return (int)cudaFree(p); }
int host_malloc(void **p, size_t s, unsigned int f) { return (int)cudaHostAlloc(p, s, f); }
int host_free(void *p) { return (int)cudaFreeHost(p); }

LLImageJ2CImpl* fallbackCreateLLImageJ2CImpl()
{
	return new LLImageJ2CnvJPEG2k();
}

std::string LLImageJ2CnvJPEG2k::getEngineInfo() const
{
	cudaDeviceProp props;
	int dev = 0; // TODO: this is currently hard coded to read the first cuda device
	cudaGetDevice(&dev);
	cudaGetDeviceProperties(&props, dev);
	return llformat("nvJPEG2k library: %d.%d, CC: %d.%d",NVJPEG2K_VER_MAJOR,NVJPEG2K_VER_MINOR, props.major, props.minor);
}

namespace
{
	bool allocate_output_buffers(nvjpeg2kImage_t& output_image, nvjpeg2kImageInfo_t& image_info, std::vector<nvjpeg2kImageComponentInfo_t> image_comp_info, int bytes_per_element, int rgb_output)
	{
		output_image.num_components = image_info.num_components;
		if(rgb_output)
		{
			// for RGB output all component outputs dimensions are equal
			for(uint32_t c = 0; c < image_info.num_components;c++)
			{
				CHECK_CUDA(cudaMallocPitch(&output_image.pixel_data[c], &output_image.pitch_in_bytes[c], image_info.image_width * bytes_per_element, image_info.image_height));
			}
		}
		else
		{
			for(uint32_t c = 0; c < image_info.num_components;c++)
			{
				CHECK_CUDA(cudaMallocPitch(&output_image.pixel_data[c], &output_image.pitch_in_bytes[c], image_comp_info[c].component_width * bytes_per_element, image_comp_info[c].component_height));
			}
		}
		return true;
	}

	inline S32 extractLong4( U8 const *aBuffer, int nOffset )
	{
		S32 ret = aBuffer[ nOffset ] << 24;
		ret += aBuffer[ nOffset + 1 ] << 16;
		ret += aBuffer[ nOffset + 2 ] << 8;
		ret += aBuffer[ nOffset + 3 ];
		return ret;
	}

	inline S32 extractShort2( U8 const *aBuffer, int nOffset )
	{
		S32 ret = aBuffer[ nOffset ] << 8;
		ret += aBuffer[ nOffset + 1 ];

		return ret;
	}

	inline bool isSOC( U8 const *aBuffer )
	{
		return aBuffer[ 0 ] == 0xFF && aBuffer[ 1 ] == 0x4F;
	}

	inline bool isSIZ( U8 const *aBuffer )
	{
		return aBuffer[ 0 ] == 0xFF && aBuffer[ 1 ] == 0x51;
	}

	bool getMetadataFast( LLImageJ2C &aImage, S32 &aW, S32 &aH, S32 &aComps )
	{
		const int J2K_HDR_LEN( 42 );
		const int J2K_HDR_LSIZ( 4 );
		const int J2K_HDR_X1( 8 );
		const int J2K_HDR_Y1( 12 );
		const int J2K_HDR_X0( 16 );
		const int J2K_HDR_Y0( 20 );
		const int J2K_HDR_NUMCOMPS( 40 );

		if( aImage.getDataSize() < J2K_HDR_LEN )
			return false;

		U8 const* pBuffer = aImage.getData();

		if( !isSOC( pBuffer ) || !isSIZ( pBuffer+2 ) )
			return false;

		// Image and tile size length
		S32 Lsiz = extractShort2( pBuffer, J2K_HDR_LSIZ );

		// Image size
		S32 Xsiz = extractLong4( pBuffer, J2K_HDR_X1 );
		S32 Ysiz = extractLong4( pBuffer, J2K_HDR_Y1 );
		S32 XOsiz = extractLong4( pBuffer, J2K_HDR_X0 );
		S32 YOsiz = extractLong4( pBuffer, J2K_HDR_Y0 );

		// Tile size
		S32 XTsiz = extractLong4( pBuffer, J2K_HDR_Y0 );
		S32 YTsiz = extractLong4( pBuffer, J2K_HDR_Y0 );
		S32 XTOsiz = extractLong4( pBuffer, J2K_HDR_Y0 );
		S32 YTOsiz = extractLong4( pBuffer, J2K_HDR_Y0 );

		S32 numComps = extractShort2( pBuffer, J2K_HDR_NUMCOMPS );

		const int J2K_HDR_COMPS( 40 );

		aComps = numComps;
		aW = Xsiz - XOsiz;
		aH = Ysiz - YOsiz;

		return true;
	}



	inline U32 readU32( U8 const *aBuffer, size_t &offset )
	{
		U32 ret = aBuffer[ offset ] << 24;
		ret += aBuffer[ offset + 1 ] << 16;
		ret += aBuffer[ offset + 2 ] << 8;
		ret += aBuffer[ offset + 3 ];
		offset += 4;
		return ret;
	}

	inline U16 readU16( U8 const *aBuffer, size_t &offset )
	{
		U16 ret = aBuffer[ offset ] << 8;
		ret += aBuffer[ offset + 1 ];
		offset += 2;
		return ret;
	}

	inline U8 readU8( U8 const *aBuffer, size_t &offset )
	{
		U8 ret = aBuffer[ offset ];
		offset += 1;
		return ret;
	}

	struct jpeg200_header_component_t
	{
		U8 Ssiz;
		U8 XRsiz;
		U8 YRsiz;
	};

	struct jpeg200_header_t
	{
		//SIZ
		U16 Lsiz;		// 38 + (Csiz * 3) 47 for 3 components, 50 for 4, 53 for 5
		U16 Rsiz;		// Should always be 0x0
		U32 Xsiz;
		U32 Ysiz;
		U32 XOsiz;
		U32 YOsiz;
		U32 XTsiz;
		U32 YTsiz;
		U32 XTOsiz;
		U32 YTOsiz;
		U16 Csiz;
		jpeg200_header_component_t Components[5]; // Some textures are silly and have 5 instead of 3/4
		U32 width;	// Inferred
		U32 height; 	// Inferred

		// COD
		U16 Lcod;		// Usually 12
		U16 Scod;
		U8 SPcod_decomposition_levels;
		U8 SPcod_progression_order;
		U16 SPcod_pumber_of_layersr;
		U8 SPcod_code_block_size_width;
		U8 SPcod_code_block_size_height;
		U8 SPcod_code_block_style;
		U8 SPcod_transform;
		U8 SPcod_multiple_component_transform;
		// packet_partition_size would go here

		// COC
		U16 Lcoc;		// 9 - 65535

		// RGN
		U16 Lrgn;		// 5 - 6

		//QCD
		U16 Lqcd;		// 4 - 197
		U8 Sqcd;

		// QCC
		U16 Lqcc;		// 5 - 199

		// POD
		U16 Lpod;		// 9 - 65535

		// TLM
		U16 Ltlm;		// 6 - 65535

		// PLM
		U16 Lplm;		// 5 - 65535

		// PLT
		U16 Lplt;		// 4 - 65535

		// PPM
		U16 Lppm;	// 6 - 65535

		// PPT
		U16 Lppt;		// 4 - 65535

		// SOP
		U16 Lsop;		// 4
		U16 Nsop;

		// CME
		U16 Lcme;	// 5 - 65535
		U16 Rcme;
		std::string Ccme;

		// SOT
		U16 Lsot;		// 10
		U16 Isot;
		U32 Psot;
		U8 TPsot;
		U8 TNsot;

		// SOD
		U16 Lsod;		// variable

		bool valid;
	};

	bool parseJPEG200Element( U8 const *pBuffer, size_t &offset, size_t &buffer_size, jpeg200_header_t &header )
	{
		U16 element_marker = readU16( pBuffer, offset );
		U16 element_length = readU16( pBuffer, offset );

		// SIZ -- required, length: 42 - 49191
		switch (element_marker)
		{
			case 0xFF51:
				header.Lsiz = element_length;
				header.Rsiz = readU16( pBuffer, offset );
				if (header.Rsiz != 0x0) // Not really necessary
				{
					 LL_WARNS("nvJPEG2k") << "header decode failed: Rsiz not 0x0, was: "
						<< std::hex << header.Rsiz << std::dec << LL_ENDL;
				}
				header.Xsiz = readU32( pBuffer, offset );
				header.Ysiz = readU32( pBuffer, offset );
				header.XOsiz = readU32( pBuffer, offset );
				header.YOsiz = readU32( pBuffer, offset );
				header.XTsiz = readU32( pBuffer, offset );
				header.YTsiz = readU32( pBuffer, offset );
				header.XTOsiz = readU32( pBuffer, offset );
				header.YTOsiz = readU32( pBuffer, offset );
				header.Csiz = readU16( pBuffer, offset );
				for(U16 comp = 0; comp < header.Csiz; comp++)
				{
					header.Components[comp].Ssiz = readU8( pBuffer, offset );
					header.Components[comp].XRsiz = readU8( pBuffer, offset );
					header.Components[comp].YRsiz = readU8( pBuffer, offset );
				}

				header.width = header.Xsiz - header.XOsiz;
				header.height = header.Ysiz - header.YOsiz;
				break;

			// COD -- required, length: 12 - 65535
			case 0xFF52:
				header.Lcod = element_length;
				header.Scod = readU8( pBuffer, offset );
				header.SPcod_decomposition_levels = readU8( pBuffer, offset );
				header.SPcod_progression_order = readU8( pBuffer, offset );
				header.SPcod_pumber_of_layersr = readU16( pBuffer, offset );
				header.SPcod_code_block_size_width = readU8( pBuffer, offset );
				header.SPcod_code_block_size_height = readU8( pBuffer, offset );
				header.SPcod_code_block_style = readU8( pBuffer, offset );
				header.SPcod_transform = readU8( pBuffer, offset );
				header.SPcod_multiple_component_transform = readU8( pBuffer, offset );
				offset += element_length - 12; // For SL textures, this should likely be zero
				// break;
				return true; // stop header decoding after finding COD

			// COC -- optional, length: 9 - 65535
			case 0xFF53:
				header.Lcoc = element_length;
				// LL_INFOS("nvJPEG2k") << "header has COC: offset=" << offset - 4
					// << " Lcoc=" << (int)element_length << LL_ENDL;
				offset += element_length - 2; // Skip COC
				break;

			// RGN -- optional, length: 5 - 6
			case 0xFF5E:
				header.Lrgn = element_length;
				// LL_INFOS("nvJPEG2k") << "header has RGN: offset=" << offset - 4
					// << " Lrgn=" << (int)element_length << LL_ENDL;
				offset += element_length - 2; // Skip RGN
				break;

			//QCD -- required, length: 4 - 197
			case 0xFF5C:
				header.Lqcd = element_length;
				header.Sqcd = readU8( pBuffer, offset );
				offset += element_length - 3;
				break;

			// QCC -- optional, length: 5 - 199
			case 0xFF5D:
				header.Lqcc = element_length;
				// LL_INFOS("nvJPEG2k") << "header has QCC: offset=" << offset - 4
					// << " Lqcc=" << (int)element_length << LL_ENDL;
				offset += element_length - 2; // Skip QCC
				break;

			// POD -- optional, length: 9 - 65535
			case 0xFF5F:
				header.Lpod = element_length;
				// LL_INFOS("nvJPEG2k") << "header has POD: offset=" << offset - 4
					// << " Lpod=" << (int)element_length << LL_ENDL;
				offset += element_length - 2; // Skip POD
				break;

			// TLM -- optional, length: 6 - 65535
			case 0xFF55:
				header.Ltlm = element_length;
				// LL_INFOS("nvJPEG2k") << "header has TLM: offset=" << offset - 4
					// << " Ltlm=" << (int)element_length << LL_ENDL;
				offset += element_length - 2; // Skip TLM
				break;

			// PLM -- optional, length: 5 - 65535
			case 0xFF57:
				header.Lplm = element_length;
				// LL_INFOS("nvJPEG2k") << "header has PLM: offset=" << offset - 4
					// << " Lplm=" << (int)element_length << LL_ENDL;
				offset += element_length - 2; // Skip PLM
				break;

			// PLT -- optional, length: 4 - 65535
			case 0xFF58:
				header.Lplt = element_length;
				// LL_INFOS("nvJPEG2k") << "header has PLT: offset=" << offset - 4
					// << " Lplt=" << (int)element_length << LL_ENDL;
				offset += element_length - 2; // Skip PLT
				break;

			// PPM -- optional, length: 6 - 65535
			case 0xFF60:
				header.Lppm = element_length;
				// LL_INFOS("nvJPEG2k") << "header has PPM: offset=" << offset - 4
					// << " Lppm=" << (int)element_length << LL_ENDL;
				offset += element_length - 2; // Skip PPM
				break;

			// PPT -- optional, length: 4 - 65535
			case 0xFF61:
				header.Lppt = element_length;
				// LL_INFOS("nvJPEG2k") << "header has PPT: offset="
					// << offset - 4 << " Lppt=" << (int)element_length << LL_ENDL;
				offset += element_length - 2; // Skip PPT
				break;

			// SOP -- optional, length: 4
			case 0xFF91:
				header.Lsop = element_length;
				header.Nsop = readU16( pBuffer, offset );
				// LL_INFOS("nvJPEG2k") << "header has SOP: offset=" << offset - 4
					// << " Lsop=" << (int)element_length
					// << " Nsop=" << (int)header.Nsop << LL_ENDL;
				break;

			// EPH -- optional, length: 0
			case 0xFF92:
				// LL_INFOS("nvJPEG2k") << "header has EPH: offset=" << offset - 4 << LL_ENDL;
				offset -= 2; // EPH does not have a length
				break;

			// CME -- optional, length:  5 - 65535
			// TODO: there may be more than one
			case 0xFF64:
				header.Lcme = element_length;
				header.Rcme = readU16( pBuffer, offset ); // Registration value of the marker segment
				if (header.Rcme == 0x1)
					header.Ccme.assign((char*)(pBuffer + offset), element_length - 4);
				LL_INFOS("nvJPEG2k") << "header has CME: offset=" << offset - element_length
					<< " Lcme=" << (int)element_length
					<< " Rcme=0x" << std::hex << header.Rcme << std::dec
					<< " Ccme=" << header.Ccme << LL_ENDL;
				offset += element_length - 4;
				break;

			// SOT -- optional, length: 10
			case 0xFF90:
				header.Lsot = element_length;
				header.Isot = readU16( pBuffer, offset );
				header.Psot = readU32( pBuffer, offset );
				header.TPsot = readU8( pBuffer, offset );
				header.TNsot = readU8( pBuffer, offset );
				// LL_INFOS("nvJPEG2k") << "header has SOT: offset=" << offset - 4
					// << " Lsot=" << (int)element_length
					// << " Isot=" << (int)header.Isot
					// << LL_ENDL;
				break;

			// SOD -- optional, length: 10
			case 0xFF93:
				element_length = 0;
				while (offset < buffer_size)
				{
					if (element_marker == 0xFF90 || element_marker == 0xFFD9)
					{
						offset -= 2; // back off so next parse pass works on this marker
						break;
					}
					element_length++;
					offset -= 1; // next maker should be in some unit of 8bits away
					element_marker = readU16( pBuffer, offset );
				}
				header.Lsod = element_length; // not accurate
				// LL_INFOS("nvJPEG2k") << "header has SOD: offset=" << offset - 4
					// << " Lsod=" << (int)element_length
					// << " Next maker=" << std::hex << element_marker << std::dec << LL_ENDL;
				break;

			// EOC -- optional, length: 0
			case 0xFFD9:
				// LL_INFOS("nvJPEG2k") << "header has EOC: offset=" << offset - 4 << LL_ENDL;
				offset -= 2; // EOC does not have a length
				break;

			// this is very hack
			default:
				// LL_INFOS("nvJPEG2k") << "header parsing encountered (currently) unsupported element marker: 0x"
				// << std::hex << element_marker << std::dec
				// << " offset=" << offset - 4 << LL_ENDL;
				offset += element_length - 2; // Skip unknown
		}

		return false;
	}

	// JPEG200 header parser
	// References used:
	// https://www.ics.uci.edu/~dan/class/267/papers/jpeg2000.pdf
	// https://www.itu.int/rec/dologin_pub.asp?lang=s&id=T-REC-T.813-201206-I!!PDF-E&type=items
	jpeg200_header_t parseJPEG200Header( LLImageJ2C &aImage )
	{
		U8 const* pBuffer = aImage.getData();
		size_t buffer_size =  aImage.getDataSize();
		size_t offset = 0;

		jpeg200_header_t header;
		header.valid = true;

		// Start of codestream -- first 16bits, required
		U16 SOC = readU16( pBuffer, offset );
		if (SOC != 0xFF4F)
		{
			 LL_WARNS("nvJPEG2k") << "header decode failed: "
			 << " SOC=0x" << std::hex << SOC
			 << std::dec << LL_ENDL;
			 header.valid = false;
			return header;
		}

		bool done;
		while (offset < buffer_size)
		{
			done = parseJPEG200Element( pBuffer, offset, buffer_size, header );
			if (!done)
				break;
		}

		return header;
	}
}


LLImageJ2CnvJPEG2k::LLImageJ2CnvJPEG2k() : LLImageJ2CImpl(),
	nvjpeg2k_decode_state(),
	nvjpeg2k_handle(),
	stream(),
	jpeg2k_stream(),
	decode_params(),
	output_image(),
	decode_output_u16(),
	decode_output_u8(),
	image_comp_info(),
	decode_output_pitch(),
	mRGB_output(1),
	startEvent(NULL),
	stopEvent(NULL)
{
	bool binit_nvJPEG2k = init_nvJPEG2k();
	if (!binit_nvJPEG2k)
	{
		LL_WARNS("nvJPEG2k") << "Could not intialize" << LL_ENDL;
	}
}

LLImageJ2CnvJPEG2k::~LLImageJ2CnvJPEG2k()
{
	bool bdestroy_nvJPEG2k = destroy_nvJPEG2k();
	if (!bdestroy_nvJPEG2k)
	{
		LL_WARNS("nvJPEG2k") << "Could not destruct" << LL_ENDL;
	}
}

bool LLImageJ2CnvJPEG2k::init_nvJPEG2k()
{
	nvjpeg2kDeviceAllocator_t dev_allocator = {&dev_malloc, &dev_free};
	nvjpeg2kPinnedAllocator_t pinned_allocator = {&host_malloc, &host_free};
	CHECK_NVJPEG2K(nvjpeg2kCreate(NVJPEG2K_BACKEND_DEFAULT, &dev_allocator, &pinned_allocator, &nvjpeg2k_handle));
	CHECK_NVJPEG2K(nvjpeg2kDecodeStateCreate(nvjpeg2k_handle, &nvjpeg2k_decode_state));
	CHECK_NVJPEG2K(nvjpeg2kStreamCreate(&jpeg2k_stream));
	CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
	CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));
	CHECK_NVJPEG2K(nvjpeg2kDecodeParamsCreate(&decode_params));
#if (NVJPEG2K_VER_MAJOR == 0 && NVJPEG2K_VER_MINOR >= 3)
	// 420 and 422 subsampling are enabled in nvJPEG2k v 0.3.0
	CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetRGBOutput(decode_params, mRGB_output));
#endif

	return true;
}

bool LLImageJ2CnvJPEG2k::destroy_nvJPEG2k()
{
	CHECK_NVJPEG2K(nvjpeg2kDecodeParamsDestroy(decode_params));
	CHECK_CUDA(cudaEventDestroy(startEvent));
	CHECK_CUDA(cudaEventDestroy(stopEvent));
	CHECK_CUDA(cudaStreamDestroy(stream));
	CHECK_NVJPEG2K(nvjpeg2kStreamDestroy(jpeg2k_stream));
	CHECK_NVJPEG2K(nvjpeg2kDecodeStateDestroy(nvjpeg2k_decode_state));
	CHECK_NVJPEG2K(nvjpeg2kDestroy(nvjpeg2k_handle));

	return true;
}

bool LLImageJ2CnvJPEG2k::initDecode(LLImageJ2C &base, LLImageRaw &raw_image, int discard_level, int* region)
{
	return false;
}

bool LLImageJ2CnvJPEG2k::initEncode(LLImageJ2C &base, LLImageRaw &raw_image, int blocks_size, int precincts_size, int levels)
{
	return false;
}

bool LLImageJ2CnvJPEG2k::decodeImpl(LLImageJ2C &base, LLImageRaw &raw_image, F32 decode_time, S32 first_channel, S32 max_channel_count)
{
	base.updateRawDiscardLevel();
	int discard_level = base.getDiscardLevel();
	int discard = (discard_level != -1 ? discard_level : base.getRawDiscardLevel());
	bool sucess = decode(
		base,
		raw_image,
		decode_time,
		first_channel,
		max_channel_count,
		discard_level,
		discard
	);

	// Debugging for failing textures
	// TODO: fallback to some other decoder
	if (!sucess)
	{
		jpeg200_header_t header = parseJPEG200Header( base );
		if (header.valid)
		{
			base.setSize(header.width, header.height, header.Csiz);
		}
		else
		{
			S32 width(0); S32 height(0); S32 img_components(0);
			if ( getMetadataFast( base, width, height, img_components ) )
			{
				base.setSize(width, height, img_components);
			}
		}

		LL_WARNS("nvJPEG2k")
		<< "\nTexture failed to load: "
		<< " discard used: " << discard << ", asked: " << discard_level
		<< " data size: " << base.getDataSize()

		<< "\nLsiz: " << header.Lsiz
		<< " XYsiz: " << header.Xsiz << "x" << header.Ysiz
		<< " XOsiz: " << header.XOsiz << "x" << header.YOsiz
		<< " XYTsiz: " << header.XTsiz << "x" << header.YTsiz
		<< " XYTOsiz: " << header.XTOsiz << "x" << header.YTOsiz
		<< " Csiz: " << header.Csiz
		<< " size: " << header.width << "x" << header.height
		<< " valid: " << header.valid

		<< "\nLcod: " << (int)header.Lcod
		<< " Scod: 0x" << std::hex << (int)header.Scod << std::dec 
		<< " decomposition_levels: 0x" << std::hex << (int)header.SPcod_decomposition_levels << std::dec 
		<< " progression_order: 0x" << std::hex << (int)header.SPcod_progression_order << std::dec 
		<< " pumber_of_layersr: 0x" << std::hex << (int)header.SPcod_pumber_of_layersr << std::dec 
		<< " code_block_size_width: " << (int)header.SPcod_code_block_size_width
		<< " code_block_size_height: " << (int)header.SPcod_code_block_size_height
		<< " code_block_style: 0x" << std::hex << (int)header.SPcod_code_block_style << std::dec 
		<< " transform: 0x" << std::hex << (int)header.SPcod_transform << std::dec 
		<< " multiple_component_transform: 0x" << std::hex << (int)header.SPcod_multiple_component_transform << std::dec 
		<< LL_ENDL;

		base.setLastError("decode failed");
		base.decodeFailed();
		return true;
	}

	// LL_INFOS("nvJPEG2k") << " Finished " << LL_ENDL;

	return true;
}


bool LLImageJ2CnvJPEG2k::decode(LLImageJ2C &base, LLImageRaw &raw_image, F32 decode_time, S32 first_channel, S32 max_channel_count, int discard_level, int discard)
{
	if (!nvjpeg2k_decode_state)
	{
		LL_WARNS("nvJPEG2k") << " nvjpeg2k_decode_state null" << LL_ENDL;
		return false;
	}
	if (!nvjpeg2k_handle)
	{
		LL_WARNS("nvJPEG2k") << " nvjpeg2k_handle null" << LL_ENDL;
		return false;
	}
	if (!stream)
	{
		LL_WARNS("nvJPEG2k") << " stream null" << LL_ENDL;
		return false;
	}
	if (!jpeg2k_stream)
	{
		LL_WARNS("nvJPEG2k") << " jpeg2k_stream null" << LL_ENDL;
		return false;
	}
	
	nvjpeg2kImageInfo_t image_info;
	int bytes_per_element = 0;
	float loopTime = 0;

	// decode start
	CHECK_CUDA(cudaStreamSynchronize(stream));
	CHECK_CUDA(cudaEventRecord(startEvent, stream));

	double parse_time = Wtime();
	CHECK_NVJPEG2K(nvjpeg2kStreamParse(nvjpeg2k_handle, (unsigned char*)base.getData(), base.getDataSize(), 0, 0, jpeg2k_stream));
	parse_time = Wtime() - parse_time;

	CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(jpeg2k_stream, &image_info));

	uint32_t num_res_levels;
	CHECK_NVJPEG2K(nvjpeg2kStreamGetResolutionsInTile(jpeg2k_stream, 0, &num_res_levels));

	// Set the base dimensions
	base.setSize(image_info.image_width, image_info.image_height, image_info.num_components);
	base.setLevels(num_res_levels);

	image_comp_info.resize(image_info.num_components);
	for (uint32_t c = 0; c < image_info.num_components; c++)
	{
		CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(jpeg2k_stream, &image_comp_info[c], c));
	}

	// SL textures should only ever be 8-bit
	decode_output_pitch.resize(image_info.num_components);
	output_image.pitch_in_bytes = decode_output_pitch.data();
	if (image_comp_info[0].precision > 8 && image_comp_info[0].precision <= 16)
	{
		decode_output_u16.resize(image_info.num_components);
		output_image.pixel_data = (void **)decode_output_u16.data();
		output_image.pixel_type = NVJPEG2K_UINT16;
		bytes_per_element = 2;
		LL_INFOS("nvJPEG2k") << "pixel_type: NVJPEG2K_UINT16" << LL_ENDL;
	}
	else if (image_comp_info[0].precision == 8)
	{
		decode_output_u8.resize(image_info.num_components);
		output_image.pixel_data = (void **)decode_output_u8.data();
		output_image.pixel_type = NVJPEG2K_UINT8;
		bytes_per_element = 1;
		// LL_INFOS("nvJPEG2k") << "pixel_type: NVJPEG2K_UINT8" << LL_ENDL;
	}
	else
	{
		LL_WARNS("nvJPEG2k") << "Precision value " << std::to_string(image_comp_info[0].precision) << " not supported" << LL_ENDL;
		return false;
	}

	if(!allocate_output_buffers(output_image, image_info, image_comp_info, bytes_per_element, mRGB_output))
	{
		return false;
	}

#if (NVJPEG2K_VER_MAJOR == 0 && NVJPEG2K_VER_MINOR >= 3)
	CHECK_NVJPEG2K(nvjpeg2kDecodeTile(
		nvjpeg2k_handle,
        nvjpeg2k_decode_state,
        jpeg2k_stream,
        decode_params,
        0,
        discard,
        &output_image,
        stream
	));
#else
	CHECK_NVJPEG2K(nvjpeg2kDecode(nvjpeg2k_handle, nvjpeg2k_decode_state, jpeg2k_stream, &output_image, stream));
#endif

	CHECK_CUDA(cudaStreamSynchronize(stream));
	CHECK_CUDA(cudaEventRecord(stopEvent, stream));
	CHECK_CUDA(cudaEventSynchronize(stopEvent));

	CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
	decode_time += static_cast<F32>(loopTime/1000.0); // loopTime is in milliseconds
	decode_time += parse_time;

	S32 channels = image_info.num_components - first_channel;
	channels = llmin(channels,max_channel_count);

	raw_image.resize(image_info.image_width, image_info.image_height, channels);
	U8 *rawp = raw_image.getData();

	S32 offset;
	unsigned char vchan[image_info.image_height * image_info.image_width * bytes_per_element];

	for (S32 comp = first_channel, dest=0; comp < first_channel + channels;
		comp++, dest++)
	{
		CHECK_CUDA(cudaMemcpy2D(
			vchan,
			(size_t)image_info.image_width * sizeof(unsigned char),
			(unsigned short *)output_image.pixel_data[comp],
			output_image.pitch_in_bytes[comp],
			image_info.image_width * sizeof(unsigned char),
			image_info.image_height,
			cudaMemcpyDeviceToHost
		));

		offset = dest;
		for (S32 y = (image_info.image_height - 1); y >= 0; y--)
		{
			for (S32 x = 0; x < image_info.image_width; x++)
			{
				rawp[offset] = vchan[y * image_info.image_width + x];
				offset += channels;
			}
		}
	}

	for(uint32_t c = 0; c < output_image.num_components; c++)
	{
		 CHECK_CUDA(cudaFree(output_image.pixel_data[c]));
	}

	return true;
}

bool LLImageJ2CnvJPEG2k::encodeImpl(LLImageJ2C &base, const LLImageRaw &raw_image, const char* comment_text, F32 encode_time, bool reversible)
{
	return true;
}


bool LLImageJ2CnvJPEG2k::getMetadata(LLImageJ2C &base)
{
	// Update the raw discard level
	base.updateRawDiscardLevel();

	jpeg200_header_t header = parseJPEG200Header(base);
	if (header.valid)
	{
		base.setSize(header.width, header.height, header.Csiz);
		return true;
	}
	else
	{
		LL_INFOS("nvJPEG2k") << "Header decode failed, falling back to getMetadataFast" << LL_ENDL;
		S32 width(0); S32 height(0); S32 img_components(0);
		if ( getMetadataFast( base, width, height, img_components ) )
		{
			base.setSize(width, height, img_components);
			return true;
		}
	}

	LL_INFOS("nvJPEG2k") << " getMetadataFast failed" << LL_ENDL;
	return false;
}
