package localembedder

import (
	"image"
	"math"

	"golang.org/x/image/draw"
)

// resizeShortestEdge resizes src so its shorter side equals edge pixels,
// preserving aspect ratio, using a cubic-convolution kernel (the closest
// available match in golang.org/x/image/draw to the reference
// preprocessor_config.json's bicubic resample — not claimed to be bit-exact,
// see the plan's Global Constraints for why that's an accepted limitation).
func resizeShortestEdge(src image.Image, edge int) *image.RGBA {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 {
		w = 1
	}
	if h <= 0 {
		h = 1
	}
	var newW, newH int
	if w < h {
		newW = edge
		newH = int(math.Round(float64(h) * float64(edge) / float64(w)))
	} else {
		newH = edge
		newW = int(math.Round(float64(w) * float64(edge) / float64(h)))
	}
	if newW < edge {
		newW = edge
	}
	if newH < edge {
		newH = edge
	}
	dst := image.NewRGBA(image.Rect(0, 0, newW, newH))
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, b, draw.Src, nil)
	return dst
}

// centerCrop returns a size x size crop centered in src. src is assumed to
// be at least size x size on both axes (resizeShortestEdge guarantees this).
func centerCrop(src *image.RGBA, size int) *image.RGBA {
	b := src.Bounds()
	x0 := b.Min.X + (b.Dx()-size)/2
	y0 := b.Min.Y + (b.Dy()-size)/2
	dst := image.NewRGBA(image.Rect(0, 0, size, size))
	draw.Draw(dst, dst.Bounds(), src, image.Point{X: x0, Y: y0}, draw.Src)
	return dst
}

// clipPreprocess resizes+center-crops img to size x size and returns a flat
// NCHW (channel-major: all R, then all G, then all B) float32 slice
// normalized per-channel with mean/std, rescaled from [0,255] to [0,1]
// first — matching the real preprocessor_config.json pipeline (do_resize,
// do_center_crop, do_rescale, do_normalize, in that order).
func clipPreprocess(img image.Image, size int, mean, std [3]float32) []float32 {
	resized := resizeShortestEdge(img, size)
	cropped := centerCrop(resized, size)

	out := make([]float32, 3*size*size)
	plane := size * size
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			r, g, b, _ := cropped.At(x, y).RGBA()
			idx := y*size + x
			out[0*plane+idx] = (float32(r>>8)/255 - mean[0]) / std[0]
			out[1*plane+idx] = (float32(g>>8)/255 - mean[1]) / std[1]
			out[2*plane+idx] = (float32(b>>8)/255 - mean[2]) / std[2]
		}
	}
	return out
}
