

#include "linden_common.h"
#include "llimagej2cjasper.h"

#include <jasper.h>

#if LL_WINDOWS
#pragma warning(disable:4189)
#endif

struct JasperData
{
	jas_image_t *mImage = nullptr;
	jas_context_t mContext;
};

LLImageJ2CImpl* fallbackCreateLLImageJ2CImpl()
{
	return new LLImageJ2CJasper();
}

std::string LLImageJ2CJasper::getEngineInfo() const
{
	return std::string("JasPer runtime: ") + jas_getversion();
}

namespace 
{
	const int LOG_BUFFER_SIZE = 1024;
	int jasperLog(jas_logtype_t aType, const char *fmt, va_list args)
	{
        // Just ignore this one as it comes up quite frequently but seems to be harmless for SL textures. Would be nice if it told us which 'tag type'
		if (strncmp(fmt, "warning: skipping unknown tag type", sizeof("warning: skipping unknown tag type")-1) == 0)
			return 0;

		char *pBufffer = new char[LOG_BUFFER_SIZE];
		int res = vsnprintf(pBufffer, LOG_BUFFER_SIZE, fmt, args);

        if( aType == JAS_LOGTYPE_CLASS_WARN )
        {
			LL_WARNS("jasper") << pBufffer << LL_ENDL;
        }
		else if ( aType == JAS_LOGTYPE_CLASS_DEBUG)
		{
			LL_DEBUGS("jasper") << pBufffer << LL_ENDL;
		}
		else
		{
			LL_INFOS("jasper") << pBufffer << LL_ENDL;
		}

		delete[] pBufffer;


		return res;
	}

	bool initJasper()
	{
		jas_conf_clear();
		jas_conf_set_multithread(1);
		jas_conf_set_vlogmsgf(jasperLog);
		if (jas_init_library())
			return false;

		return true;
	}

}

LLImageJ2CJasper::LLImageJ2CJasper()
	: LLImageJ2CImpl()
{
    static bool bJasperInit{ initJasper() };
    if( !bJasperInit )
    {
        LL_WARNS("Jasper") << "Could not intialize" << LL_ENDL;
    }

    mData = std::make_unique<JasperData>();
    mData->mContext = jas_context_create();
}


LLImageJ2CJasper::~LLImageJ2CJasper()
{
	if (mData->mImage)
		jas_image_destroy(mData->mImage);

	jas_context_destroy(mData->mContext);
}

bool LLImageJ2CJasper::initDecode(LLImageJ2C &base, LLImageRaw &raw_image, int discard_level, int* region)
{
	return false;
}

bool LLImageJ2CJasper::initEncode(LLImageJ2C &base, LLImageRaw &raw_image, int blocks_size, int precincts_size, int levels)
{
	return false;
}

bool LLImageJ2CJasper::decodeImpl(LLImageJ2C &base, LLImageRaw &raw_image, F32 decode_time, S32 first_channel, S32 max_channel_count)
{
	if (!mData->mImage)
		return false;

	jas_set_context(mData->mContext);
	U32 numcmpts = jas_image_numcmpts(mData->mImage);
	auto width = jas_image_cmptwidth(mData->mImage, 0);
	auto height = jas_image_cmptheight(mData->mImage, 0);

	numcmpts = std::min(numcmpts, (U32)max_channel_count);

	if (width*height*numcmpts == 0)
		return false;

	if (!raw_image.resize(width, height, numcmpts))
		return false;

	U8 *rawp = raw_image.getData();
	U8 *p{ rawp };

	if (numcmpts == 4)
	{
		jas_matrix_t* pixelsr = jas_matrix_create(1, width);
		jas_matrix_t* pixelsg = jas_matrix_create(1, width);
		jas_matrix_t* pixelsb = jas_matrix_create(1, width);
		jas_matrix_t* pixelsa = jas_matrix_create(1, width);

		for (int j = 0; j < height; ++j)
		{
			int const j2cRow = height - 1 - j;
			jas_image_readcmpt(mData->mImage, 0, 0, j2cRow, width, 1, pixelsr);
			jas_image_readcmpt(mData->mImage, 1, 0, j2cRow, width, 1, pixelsg);
			jas_image_readcmpt(mData->mImage, 2, 0, j2cRow, width, 1, pixelsb);
			jas_image_readcmpt(mData->mImage, 3, 0, j2cRow, width, 1, pixelsa);

			for (int k = 0; k < width; ++k)
			{
				*p++ = jas_matrix_get(pixelsr, 0, k);
				*p++ = jas_matrix_get(pixelsg, 0, k);
				*p++ = jas_matrix_get(pixelsb, 0, k);
				*p++ = jas_matrix_get(pixelsa, 0, k);
			}
		}
		jas_matrix_destroy(pixelsr);
		jas_matrix_destroy(pixelsg);
		jas_matrix_destroy(pixelsb);
		jas_matrix_destroy(pixelsa);
	}
	else if( numcmpts == 3 )
	{
		jas_matrix_t* pixelsr = jas_matrix_create(1, width);
		jas_matrix_t* pixelsg = jas_matrix_create(1, width);
		jas_matrix_t* pixelsb = jas_matrix_create(1, width);

		for (int j = 0; j < height; ++j)
		{
			int const j2cRow = height - 1 - j;
			jas_image_readcmpt(mData->mImage, 0, 0, j2cRow, width, 1, pixelsr);
			jas_image_readcmpt(mData->mImage, 1, 0, j2cRow, width, 1, pixelsg);
			jas_image_readcmpt(mData->mImage, 2, 0, j2cRow, width, 1, pixelsb);

			for (int k = 0; k < width; ++k)
			{
				*p++ = jas_matrix_get(pixelsr, 0, k);
				*p++ = jas_matrix_get(pixelsg, 0, k);
				*p++ = jas_matrix_get(pixelsb, 0, k);
			}
		}
		jas_matrix_destroy(pixelsr);
		jas_matrix_destroy(pixelsg);
		jas_matrix_destroy(pixelsb);
	}
	else
	{
		jas_matrix_t* pixels = jas_matrix_create(1, width);

		for (int i = 0; i < numcmpts; ++i)
		{
			for (int j = 0; j < height; ++j)
			{
				jas_image_readcmpt(mData->mImage, i, 0, j, width, 1, pixels);

				U8* p = rawp + (height - 1 - j) * width*numcmpts + i;
				for (int k = 0; k < width; ++k)
				{
					*p = jas_matrix_get(pixels, 0, k);
					p += numcmpts;
				}
			}
		}
		jas_matrix_destroy(pixels);
	}

	return true;
}


bool LLImageJ2CJasper::encodeImpl(LLImageJ2C &base, const LLImageRaw &raw_image, const char* comment_text, F32 encode_time, bool reversible)
{
	return true;
}

bool LLImageJ2CJasper::getMetadata(LLImageJ2C &base)
{
	jas_set_context(mData->mContext);
	base.updateRawDiscardLevel();
	jas_stream_t * stream = jas_stream_memopen((char*)base.getData(), base.getDataSize());

	auto fmtid = jas_image_getfmt(stream);

	llassert_always(stream);
	llassert_always(fmtid >= 0);

	mData->mImage = jas_image_decode(stream, /*jas_image_strtofmt((char*)"jpc")*/ fmtid, "");
	if (!mData->mImage)
	{
		jas_stream_close(stream);
		return false;
	}

	llassert_always(mData->mImage);

	auto numcmpts = jas_image_numcmpts(mData->mImage);
	auto width = jas_image_cmptwidth(mData->mImage, 0);
	auto height = jas_image_cmptheight(mData->mImage, 0);
	base.setSize(width, height, numcmpts);
	jas_stream_close(stream);

	return true;
}
