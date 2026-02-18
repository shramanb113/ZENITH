package analysis

import "strings"

func strip(input string) string {
	input = strings.ToUpper(input)

	var results strings.Builder
	for _, r := range input {
		if 'A' <= r && r <= 'Z' {
			results.WriteRune(r)
		}
	}
	return results.String()
}

func getCode(r uint8) byte {
	switch r {
	case 'B', 'F', 'P', 'V':
		return '1'
	case 'C', 'G', 'J', 'K', 'Q', 'S', 'X', 'Z':
		return '2'
	case 'D', 'T':
		return '3'
	case 'L':
		return '4'
	case 'M', 'N':
		return '5'
	case 'R':
		return '6'
	}
	return '0'
}

func Soundex(input string) string {

	clean := strip(input)
	if len(clean) == 0 {
		return ""
	}

	res := []byte{'0', '0', '0', '0'}
	res[0] = clean[0]
	count := 1

	lastCode := getCode(clean[0])

	for i := 1; i < len(clean) && count < 4; i++ {
		currCode := getCode(clean[i])

		if currCode == '0' {
			lastCode = '0'
			continue
		}

		if currCode == lastCode {
			continue
		}

		res[count] = currCode
		lastCode = currCode
		count++
	}

	return string(res)
}
