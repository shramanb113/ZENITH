> **ZENITH fork notice:** this is a locally patched copy of
> `github.com/ledongthuc/pdf@v0.0.0-20250511090121-5959a4027728`, wired in via
> a `replace` directive in the repo root's `go.mod`. The only change is in
> `read.go`'s `applyFilter`: upstream's `Value.Reader()` panics
> (`"unknown filter X"`) for any PDF stream filter it doesn't special-case,
> which includes every image-codec filter (`DCTDecode`/JPEG,
> `CCITTFaxDecode`, `JBIG2Decode`, `JPXDecode`) — making `Reader()`
> unusable for the extremely common case of a JPEG image embedded in a PDF.
> This fork adds a pass-through for those four filter names (correct per the
> PDF spec: a stream using one of them already contains the final compressed
> image data, nothing left to unwrap) instead of panicking. Needed by
> `internal/pdf`'s embedded-image OCR feature (see CLAUDE.md /
> ROADMAP.md's R7). No upstream issue/PR filed yet as of this fork
> (2026-10-09) — re-check upstream before re-syncing a newer pinned version,
> and re-apply this same patch if it hasn't landed there.
>
> To re-sync with a newer upstream release: diff this directory against the
> new `github.com/ledongthuc/pdf@<newer-version>` module cache contents,
> re-apply the `applyFilter` patch above, update the version noted here and
> in `go.mod`'s `replace` directive.

# PDF Reader

[![Built with WeBuild](https://raw.githubusercontent.com/webuild-community/badge/master/svg/WeBuild.svg)](https://webuild.community)

A simple Go library which enables reading PDF files. Forked from https://github.com/rsc/pdf

Features
  - Get plain text content (without format)
  - Get Content (including all font and formatting information)

## Install:

`go get -u github.com/ledongthuc/pdf`

## Examples:

 - Check in examples/ folder


## Read plain text

```golang
package main

import (
	"bytes"
	"fmt"

	"github.com/ledongthuc/pdf"
)

func main() {
	pdf.DebugOn = true

	f, r, err := pdf.Open("./pdf_test.pdf")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	var buf bytes.Buffer
	b, err := r.GetPlainText()
	if err != nil {
		panic(err)
	}
	buf.ReadFrom(b)
	content := buf.String()
	fmt.Println(content)
}
```

## Read all text with styles from PDF

```golang
package main

import (
	"fmt"

	"github.com/ledongthuc/pdf"
)

func main() {
	f, r, err := pdf.Open("./pdf_test.pdf")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	sentences, err := r.GetStyledTexts()
	if err != nil {
		panic(err)
	}

	// Print all sentences
	for _, sentence := range sentences {
		fmt.Printf("Font: %s, Font-size: %f, x: %f, y: %f, content: %s \n",
			sentence.Font,
			sentence.FontSize,
			sentence.X,
			sentence.Y,
			sentence.S)
	}
}
```


## Read text grouped by rows

```golang
package main

import (
	"fmt"
	"os"

	"github.com/ledongthuc/pdf"
)

func main() {
	content, err := readPdf(os.Args[1]) // Read local pdf file
	if err != nil {
		panic(err)
	}
	fmt.Println(content)
	return
}

func readPdf(path string) (string, error) {
	f, r, err := pdf.Open(path)
	defer func() {
		_ = f.Close()
	}()
	if err != nil {
		return "", err
	}
	totalPage := r.NumPage()

	for pageIndex := 1; pageIndex <= totalPage; pageIndex++ {
		p := r.Page(pageIndex)
		if p.V.IsNull() || p.V.Key("Contents").Kind() == pdf.Null {
			continue
		}

		rows, _ := p.GetTextByRow()
		for _, row := range rows {
		    println(">>>> row: ", row.Position)
		    for _, word := range row.Content {
		        fmt.Println(word.S)
		    }
		}
	}
	return "", nil
}
```

## Demo
![Run example](https://i.gyazo.com/01fbc539e9872593e0ff6bac7e954e6d.gif)
