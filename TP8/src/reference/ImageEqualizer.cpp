#include <reference/ImageEqualizer.h>
#include <algorithm> // min/max
#include <float.h>

#ifndef FLT_EPSILON 
#  define FLT_EPSILON 1e-7f
#endif

namespace
{
    using uint = unsigned int;
    using uchar = unsigned char;
    // petites définitions (existantes sous CUDA)
    struct float3
    {
        float x, y, z;
    };
    struct uchar3
    {
        uchar x, y, z;
        uchar3(uchar x, uchar y, uchar z) : x(x), y(y), z(z) {}
        uchar3(PPMBitmap::RGBcol color) : x(color.r), y(color.g), z(color.b) {}
    };

    // conversions
    float3 RGB2HSV( const uchar3 inRGB ) 
    {
        const float R = float( inRGB.x ) / 256.f; // les valeurs sont normalisées entre 0 et 1 (exclu)
        const float G = float( inRGB.y ) / 256.f;
        const float B = float( inRGB.z ) / 256.f;

        const float min	= std::min( R, std::min( G, B ) ); // en CUDA utilisez fminf
        const float max	= std::max( R, std::max( G, B ) ); //               et fmaxf
        const float delta = max - min;

        // H
        float H;
        if		( delta < FLT_EPSILON )  
            H = 0.f;
        else if	( max == R )	
            H = 60.f * ( G - B ) / ( delta + FLT_EPSILON ) + 360.f;
        else if ( max == G )	
            H = 60.f * ( B - R ) / ( delta + FLT_EPSILON ) + 120.f;
        else					
            H = 60.f * ( R - G ) / ( delta + FLT_EPSILON ) + 240.f;
        while	( H >= 360.f )	
            H -= 360.f ;

        // S
        const float S = max < FLT_EPSILON ? 0.f : 1.f - min / max;

        // V
        const float V = max;

        return float3{ H, S, V };
    }

    uchar3 HSV2RGB( const float H, const float S, const float V ) 
    {
        const float	d	= H / 60.f; // de [0,360[ nous passons à [0, 6[ ...
        const int	hi	= int(d) % 6;
        const float f	= d - float(hi); // partie décimale

        const float l   = V * ( 1.f - S );
        const float m	= V * ( 1.f - f * S );
        const float n	= V * ( 1.f - ( 1.f - f ) * S ); 

        float R, G, B;

        if		( hi == 0 ) 
            { R = V; G = n;	B = l; }
        else if ( hi == 1 ) 
            { R = m; G = V;	B = l; }
        else if ( hi == 2 ) 
            { R = l; G = V;	B = n; }
        else if ( hi == 3 ) 
            { R = l; G = m;	B = V; }
        else if ( hi == 4 ) 
            { R = n; G = l;	B = V; }
        else				
            { R = V; G = l;	B = m; }
            
        return uchar3{ uchar(R * 256.f), uchar(G * 256.f), uchar(B * 256.f) };
    }
}


void ImageEqualizer::computeHSVfromRGB()
{
    // il suffit d'appliquer la fonction de transformation définie ci-dessus
    for(auto y=source.getHeight(); y-- > 0; )
        for(auto x=source.getWidth(); x-- > 0; )
        {
            const float3 HSV( RGB2HSV( source.getPixel(x, y) ));
            const auto offset = x + y * source.getWidth();
            Hue[offset] = HSV.x;
            Saturation[offset] = HSV.y;
            Value[offset] = HSV.z;
        }
}

void ImageEqualizer::computeHistogramFromValue()
{
    // il suffit de parcourir value ... attention à l'initialisation du résultat !
    for(auto& h : Histogram) h = 0u;
    for(auto value : Value) ++Histogram[uchar(value*256.f)];
    // check
    size_t sum = size_t(0);
    for(auto& h : Histogram) sum += size_t(h);
    if( sum != source.getWidth()*source.getHeight() )
    {
        std::cerr << "Problem in reference calculation: bad HISTOGRAM!" << std::endl;
        exit(-1);
     }
}

void ImageEqualizer::computeRepartitionFunction()
{
    Repartition[0] = Histogram[0];
    for(auto i = 1; i<256; ++i)
        Repartition[i] = Repartition[i-1] + Histogram[i];
    // check
    size_t sum = size_t(Repartition[255]);
    if( sum != source.getWidth()*source.getHeight() )
    {
        std::cerr << "Problem in reference calculation: bad REPARTITION !" << std::endl;
        exit(-1);
     }
}

void ImageEqualizer::computeFinalTransformation()
{
    // ne pas oubliez n (permet de passer d'un histogramme à une probabilité discrète ...)
    const float n = float(source.getWidth() * source.getHeight());
    for(auto i=Transformation.size(); i-- > 0;)
    {
        const auto xi = uchar(Value[i]*256.f);
        Transformation[i] = (255.f * float(Repartition[xi])) / (256.f*n);
    }
}

void ImageEqualizer::applyFinalTransformation()
{
    for(auto i=Transformation.size(); i-- > 0;)
    {
        const unsigned int x = uint(i % destination.getWidth());
        const unsigned int y = uint(i / destination.getWidth());
        const auto t = Transformation[i];
        const float3 hsv{Hue[i], Saturation[i], t};
        const uchar3 rgb(HSV2RGB(hsv.x, hsv.y, hsv.z));
        destination.setPixel(x, y, PPMBitmap::RGBcol(rgb.x, rgb.y, rgb.z));
    }
}
