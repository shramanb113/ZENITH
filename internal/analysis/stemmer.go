package analysis

import "strings"

func Stem(token string) string {

	if len(token) <= 3 {
		return token
	}

	if strings.HasSuffix(token, "ing") {
		return strings.TrimSuffix(token, "ing")
	}
	if strings.HasSuffix(token, "ed") {
		return strings.TrimSuffix(token, "ed")
	}

	if strings.HasSuffix(token, "s") && !strings.HasSuffix(token, "ss") && len(token) > 3 {
		return strings.TrimSuffix(token, "s")
	}

	return token
}
