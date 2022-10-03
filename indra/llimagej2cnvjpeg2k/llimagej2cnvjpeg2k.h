
#ifndef LL_LLImageJ2CnvJPEG2k_H
#define LL_LLImageJ2CnvJPEG2k_H

#include "linden_common.h"
#include "llimagej2c.h"

#include "nvjpeg2k.h"


const char* nvjpeg2kStatus_Strings[10] = {
	"NVJPEG2K_STATUS_SUCCESS",
	"NVJPEG2K_STATUS_NOT_INITIALIZED",
	"NVJPEG2K_STATUS_INVALID_PARAMETER",
	"NVJPEG2K_STATUS_BAD_JPEG",
	"NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED",
	"NVJPEG2K_STATUS_ALLOCATOR_FAILURE",
	"NVJPEG2K_STATUS_EXECUTION_FAILED",
	"NVJPEG2K_STATUS_ARCH_MISMATCH",
	"NVJPEG2K_STATUS_INTERNAL_ERROR",
	"NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED",
};


#define CHECK_CUDA(call)                                                                                          \
	{                                                                                                             \
		cudaError_t _e = (call);                                                                                  \
		if (_e != cudaSuccess)                                                                                    \
		{                                                                                                         \
			LL_WARNS("nvJPEG2k") << "CUDA Runtime failure: '#" << _e << "'" << LL_ENDL; \
			return false;                                                                                     \
		}                                                                                                         \
	}

#define CHECK_NVJPEG2K(call)                                                                                \
	{                                                                                                       \
		nvjpeg2kStatus_t _e = (call);                                                                       \
		if (_e != NVJPEG2K_STATUS_SUCCESS)                                                                  \
		{                                                                                                   \
			LL_WARNS("nvJPEG2k") << "NVJPEG failure: " << nvjpeg2kStatus_Strings[_e] << LL_ENDL; \
			return false;                                                                            \
		}                                                                                                   \
	}

constexpr int PIPELINE_STAGES = 10;
constexpr int MAX_PRECISION = 16;
constexpr int NUM_COMPONENTS = 4;

F32 Wtime(void)
{
#if defined(_WIN32)
	LARGE_INTEGER t;
	static F32 oofreq;
	static int checkedForHighResTimer;
	static BOOL hasHighResTimer;

	if (!checkedForHighResTimer)
	{
		hasHighResTimer = QueryPerformanceFrequency(&t);
		oofreq = 1.0 / (F32)t.QuadPart;
		checkedForHighResTimer = 1;
	}
	if (hasHighResTimer)
	{
		QueryPerformanceCounter(&t);
		return (F32)t.QuadPart * oofreq;
	}
	else
	{
		return (F32)GetTickCount() / 1000.0;
	}
#else
	struct timespec tp;
	int rv = clock_gettime(CLOCK_MONOTONIC, &tp);

	if (rv)
		return 0;

	return tp.tv_nsec / 1.0E+9 + (F32)tp.tv_sec;

#endif
}


class LLImageJ2CnvJPEG2k : public LLImageJ2CImpl
{
public:
	LLImageJ2CnvJPEG2k();
	virtual ~LLImageJ2CnvJPEG2k();
protected:
	bool init_nvJPEG2k();
	bool destroy_nvJPEG2k();

	virtual bool getMetadata(LLImageJ2C &base);
	virtual bool decodeImpl(LLImageJ2C &base, LLImageRaw &raw_image, F32 decode_time, S32 first_channel, S32 max_channel_count);
	virtual bool encodeImpl(LLImageJ2C &base, const LLImageRaw &raw_image, const char* comment_text, F32 encode_time=0.0,
								bool reversible = false);
	virtual bool initDecode(LLImageJ2C &base, LLImageRaw &raw_image, int discard_level = -1, int* region = NULL);
	virtual bool initEncode(LLImageJ2C &base, LLImageRaw &raw_image, int blocks_size = -1, int precincts_size = -1, int levels = 0);
	virtual std::string getEngineInfo() const;
	
	bool decode(LLImageJ2C &base, LLImageRaw &raw_image, F32 decode_time, S32 first_channel, S32 max_channel_count, int discard_level, int discard);

private:
	nvjpeg2kDecodeState_t nvjpeg2k_decode_state;
	nvjpeg2kHandle_t nvjpeg2k_handle;
	cudaStream_t stream;
	nvjpeg2kStream_t jpeg2k_stream;
	nvjpeg2kDecodeParams_t decode_params;
	nvjpeg2kImage_t output_image;
	std::vector<unsigned short *> decode_output_u16;
	std::vector<unsigned char *> decode_output_u8;
	std::vector<nvjpeg2kImageComponentInfo_t> image_comp_info;
	std::vector<size_t> decode_output_pitch;
	S32 mRGB_output;
	cudaEvent_t startEvent;
	cudaEvent_t stopEvent;
};

#endif
