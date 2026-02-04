package core

type StatusString string

const (
	TypePending   StatusString = "Pending"
	TypeConverted StatusString = "Converted"
)

type Document struct {
	ID       string
	Fields   map[string]string
	Vectors  map[string][]float32
	Metadata map[string]interface{}
	Version  int64
	Status   StatusString
}

func (d *Document) EstimateSize() int {
	size := 0

	size += len(d.ID)

	for k, v := range d.Fields {
		size += len(k) + len(v)
	}

	for k, v := range d.Vectors {
		size += len(k)
		size += len(v) * 4
	}

	size += 64

	size += 8
	size += len(d.Status)

	return size
}
