

#ifndef LL_LLImageJ2CJasper_H
#define LL_LLImageJ2CJasper_H

#include "llimagej2c.h"

struct JasperData;
class LLImageJ2CJasper : public LLImageJ2CImpl
{
	std::unique_ptr<JasperData> mData;
public:
	LLImageJ2CJasper();
	virtual ~LLImageJ2CJasper();
protected:
	virtual bool getMetadata(LLImageJ2C &base);
	virtual bool decodeImpl(LLImageJ2C &base, LLImageRaw &raw_image, F32 decode_time, S32 first_channel, S32 max_channel_count);
	virtual bool encodeImpl(LLImageJ2C &base, const LLImageRaw &raw_image, const char* comment_text, F32 encode_time=0.0,
								bool reversible = false);
	virtual bool initDecode(LLImageJ2C &base, LLImageRaw &raw_image, int discard_level = -1, int* region = NULL);
	virtual bool initEncode(LLImageJ2C &base, LLImageRaw &raw_image, int blocks_size = -1, int precincts_size = -1, int levels = 0);
    virtual std::string getEngineInfo() const;
};

#endif
