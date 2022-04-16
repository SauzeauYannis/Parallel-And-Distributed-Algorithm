#pragma once
#include <ppm.h>
#include <algorithm>

class ImageBlockEffect
{
    struct float3 {
        float R, G, B;
        explicit float3() : R( 0 ), G ( 0 ), B ( 0 ) {}
		explicit float3( float red, float green, float blue ) 
			: R( red ), G( green ), B( blue ) {};
		float3(const float3&) = default;
    };
    PPMBitmap& source;
    PPMBitmap& result;
    const unsigned width;
    const unsigned height;

public:
    ImageBlockEffect() = delete;
    ImageBlockEffect(PPMBitmap& source, PPMBitmap& result)
    : source(source), result(result), width(source.getWidth()), height(source.getHeight())
    {}
    ImageBlockEffect(const ImageBlockEffect&) = default;
    ImageBlockEffect& operator=(const ImageBlockEffect&) = delete;

    void applyBlockEffet()
    {
        for (unsigned row=0; row<height; row+=32)
            for(unsigned column=0; column<width; column+=32)
                fillOneImageBlock(column, row);
    }

private:
    void fillOneImageBlock(const unsigned column, const unsigned row) 
    {
        const PPMBitmap::RGBcol color = getMeanColorPerBlock(column, row);
        for(unsigned y=row; y<std::min(row+32,height); ++y)
            for(unsigned x=column; x<std::min(column+32,width); x++)
                result.setPixel(x,y, color);
    }

    PPMBitmap::RGBcol getMeanColorPerBlock(const unsigned column, const unsigned row )
    {
        // moyenne des valeurs en R, G et B ...
        float3 RGB = addAllPixelsPerBlock(column, row);
        const unsigned nbPixels = getNumberPixelsPerBlock(column, row);
        RGB.R /= float(nbPixels);
        RGB.G /= float(nbPixels);
        RGB.B /= float(nbPixels);
        return PPMBitmap::RGBcol(uchar(RGB.R),uchar(RGB.G),uchar(RGB.B));        
    }

    float3 addAllPixelsPerBlock(const unsigned column, const unsigned row)
    {
        float3 RGB;
        for(unsigned y=row; y<std::min(row+32,height); ++y)
            for(unsigned x=column; x<std::min(column+32,width); x++)
                addOnePixel(x, y, RGB);
        return RGB;
    }

    void addOnePixel(const unsigned column, const unsigned row, float3& RGB)
    {
        PPMBitmap::RGBcol color = source.getPixel(column, row);
        RGB.R += float(color.r);
        RGB.G += float(color.g);
        RGB.B += float(color.b);
    }

    unsigned getNumberPixelsPerBlock(const unsigned column, const unsigned row)
    {
        return (std::min(column+32,width)-column)*(std::min(row+32,height)-row);
    }
};