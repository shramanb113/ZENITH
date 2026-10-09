package localembedder

import (
	"image"
	"math"

	"golang.org/x/image/draw"
)

// cropRectForResizeThenCenterCrop computes, in SOURCE pixel space, the
// rectangle that "resize shortest edge to `size`, then center-crop to
// size x size" would read from — without ever materializing the
// full-resolution resized intermediate.
//
// Derivation: if w<h, resizing to shortest-edge=size makes the image `size`
// wide and h*size/w tall; the vertical center crop then takes the middle
// `size` rows of that resized image, i.e. rows
// [(h*size/w-size)/2, (h*size/w-size)/2+size). Mapping that back to source
// space by the same scale factor (source = resized / (size/w) = resized*w/size)
// gives a source-space vertical strip of height size*(w/size) = w — the full
// source width, consistent with "the short edge maps 1:1, the long edge gets
// cropped". The symmetric case (h<=w) crops horizontally instead.
func cropRectForResizeThenCenterCrop(w, h, size int) image.Rectangle {
	if w <= 0 {
		w = 1
	}
	if h <= 0 {
		h = 1
	}
	if w < h {
		// Resized image is `size` wide, `resizedH` tall.
		resizedH := int(math.Round(float64(h) * float64(size) / float64(w)))
		if resizedH < size {
			resizedH = size
		}
		// Crop window in resized space: middle `size` rows.
		cropTopResized := (resizedH - size) / 2
		// Map back to source space: scale factor resized->source is w/size.
		scale := float64(w) / float64(size)
		y0 := int(math.Round(float64(cropTopResized) * scale))
		cropH := int(math.Round(float64(size) * scale))
		if cropH < 1 {
			cropH = 1
		}
		y1 := y0 + cropH
		if y1 > h {
			y1 = h
		}
		if y0 > y1 {
			y0 = y1
		}
		return image.Rect(0, y0, w, y1)
	}

	// h <= w: resized image is `size` tall, `resizedW` wide.
	resizedW := int(math.Round(float64(w) * float64(size) / float64(h)))
	if resizedW < size {
		resizedW = size
	}
	cropLeftResized := (resizedW - size) / 2
	scale := float64(h) / float64(size)
	x0 := int(math.Round(float64(cropLeftResized) * scale))
	cropW := int(math.Round(float64(size) * scale))
	if cropW < 1 {
		cropW = 1
	}
	x1 := x0 + cropW
	if x1 > w {
		x1 = w
	}
	if x0 > x1 {
		x0 = x1
	}
	return image.Rect(x0, 0, x1, h)
}

// resizeShortestEdgeAndCenterCrop performs the equivalent of
// "resize so the shorter side equals size, then center-crop to size x size"
// in a single allocation and a single scale operation: it computes the
// corresponding crop rectangle directly in source coordinates (see
// cropRectForResizeThenCenterCrop) and scales only that region straight into
// a size x size destination, using a cubic-convolution kernel (the closest
// available match in golang.org/x/image/draw to the reference
// preprocessor_config.json's bicubic resample — not claimed to be bit-exact,
// see the plan's Global Constraints for why that's an accepted limitation).
//
// This avoids ever materializing a full-resolution resized intermediate: for
// an extreme-aspect-ratio image (e.g. 1x5000px), resizing the whole image
// first before cropping would allocate a ~size x (5000*size)px buffer
// (measured ~1.16GB for size=224) even though only a size x size result is
// ever used.
func resizeShortestEdgeAndCenterCrop(src image.Image, size int) *image.RGBA {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	cropRect := cropRectForResizeThenCenterCrop(w, h, size)
	// cropRect is relative to a (0,0)-origin view of src's dimensions;
	// translate it into src's actual bounds.
	srcCrop := cropRect.Add(b.Min)

	dst := image.NewRGBA(image.Rect(0, 0, size, size))
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, srcCrop, draw.Src, nil)
	return dst
}

// clipPreprocess resizes+center-crops img to size x size and returns a flat
// NCHW (channel-major: all R, then all G, then all B) float32 slice
// normalized per-channel with mean/std, rescaled from [0,255] to [0,1]
// first — matching the real preprocessor_config.json pipeline (do_resize,
// do_center_crop, do_rescale, do_normalize, in that order).
func clipPreprocess(img image.Image, size int, mean, std [3]float32) []float32 {
	cropped := resizeShortestEdgeAndCenterCrop(img, size)

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
