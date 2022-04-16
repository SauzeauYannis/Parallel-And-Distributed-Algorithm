#pragma once

namespace 
{
	// This functor is doing the MAP for exercise 3... 
	template<int TSIZE=3>
	struct MapFunctor 
	{
		uchar2 const*const map;
		const unsigned imageWidth;
		const unsigned imageHeight; 

		MapFunctor(
			uchar2 const*const map,
			const unsigned imageWidth,
			const unsigned imageHeight
		) :
			map(map), imageWidth(imageWidth), imageHeight(imageHeight)
		{}
		
		__device__
		size_t operator[](size_t i)
		{
			const size_t x = i % imageWidth;
			const size_t y = i / imageWidth;
			const size_t thumbnailWidth = size_t(imageWidth / TSIZE);
			const size_t xInBlock = size_t(x % thumbnailWidth);
			const size_t thumbnailX = size_t(umin(TSIZE-1, x / thumbnailWidth));
			const size_t thumbnailHeight = size_t(imageHeight / TSIZE);
			const size_t yInBlock = size_t(y % thumbnailHeight);
			const size_t thumbnailY = size_t(umin(TSIZE-1, y / thumbnailHeight));
			const uchar2 thumbnailCoord = map[thumbnailY*TSIZE + thumbnailX];
			return xInBlock + size_t(thumbnailCoord.x) * thumbnailWidth 
				+  imageWidth * (yInBlock + size_t(thumbnailCoord.y) * thumbnailHeight);
		}
	};
}