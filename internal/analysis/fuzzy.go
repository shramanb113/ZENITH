package analysis

const MAX_DISTANCE = 3

func Levenshtein(s1, s2 string) (int, bool) {

	if len(s1) > len(s2) {
		s1, s2 = s2, s1
	}

	n, m := len(s1), len(s2)

	prevRow := make([]int, n+1)
	currRow := make([]int, n+1)

	// distance from an empty string (each block)
	for i := 0; i <= n; i++ {
		prevRow[i] = i
	}

	var cost int

	for j := 1; j <= m; j++ {

		currRow[0] = j
		for i := 1; i <= n; i++ {

			cost = 1

			if s1[i-1] == s2[j-1] {
				cost = 0
			}

			// deletion , insertion , substitution
			currRow[i] = min(prevRow[i]+1, currRow[i-1]+1, prevRow[i-1]+cost)
		}

		stopCalculation := true
		for _, val := range currRow {
			if val <= MAX_DISTANCE {
				stopCalculation = false
				break
			}
		}

		if stopCalculation {
			return 0, false
		}
		copy(prevRow, currRow)
	}

	return prevRow[n], true
}
