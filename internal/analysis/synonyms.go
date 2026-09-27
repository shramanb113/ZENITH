package analysis

import (
	"bufio"
	"fmt"
	"io"
	"regexp"
	"slices"
	"strings"
	"sync"
)

// synonymMap is a static English synonym table for query expansion.
// Keys are stemmed terms (post-Porter2). Values are stemmed synonyms.
// All entries must already be stemmed — run stem() on both sides when adding.
//
// Design decisions:
//   - Static table, not a runtime-loaded file. Zero I/O at startup.
//   - Stemmed keys only — avoids double-stemming at query time.
//   - Conservative scope: only expand when semantic drift is low.
//     "ml" → "learn" is safe. "bank" → "river" is not.
//   - Used exclusively at query time, never at index time.
//     Indexing with synonyms causes index bloat and breaks IDF calculations.
var synonymMap = map[string][]string{
	// Machine learning / AI
	"ml":        {"learn", "neural", "model"},
	"ai":        {"intellig", "learn", "neural"},
	"learn":     {"train", "model", "neural"},
	"neural":    {"network", "deep", "learn"},
	"deep":      {"neural", "learn", "layer"},
	"nlp":       {"languag", "text", "process"},
	"llm":       {"languag", "model", "transform"},
	"transform": {"attent", "neural", "model"},
	"classifi":  {"label", "categor", "predict"},
	// stem("embedding")=="embed" but stem("embed")=="emb": keep both keys so
	// either surface form a user types reaches the same synonyms.
	"embed":  {"vector", "represent", "encod"},
	"emb":    {"vector", "represent", "encod"},
	"vector": {"embed", "represent", "encod"},
	"infer":     {"predict", "reason", "deduc"},
	"predict":   {"infer", "forecast", "model"},
	"train":     {"learn", "fit", "optim"},
	"optim":     {"train", "tune", "minim"},
	"fine":      {"tune", "adjust", "train"},
	"retriev":   {"search", "fetch", "lookup"},
	"generat":   {"creat", "synthes", "produc"},
	"token":     {"word", "term", "lexem"},

	// Distributed systems / infrastructure
	"distribut":   {"cluster", "parallel", "scalabl"},
	"cluster":     {"distribut", "node", "group", "segment", "similar"},
	"replicat":    {"backup", "mirror", "copi"},
	"shard":       {"partit", "split", "distribut"},
	"consensus":   {"raft", "paxo", "agreement"},
	"raft":        {"consensus", "leader", "elect"},
	"fault":       {"error", "failur", "crash"},
	"failov":      {"recov", "redundanc", "backup"},
	"latenc":      {"delay", "rtt", "respons"},
	"throughput":  {"bandwidth", "rate", "capac"},
	"scalabl":     {"distribut", "elast", "grow"},
	"microservic": {"servic", "api", "endpoint"},
	"messag":      {"queue", "event", "stream"},
	"stream":      {"flow", "event", "pipe"},
	"pipelin":     {"workflow", "process", "stream"},
	"orchestr":    {"schedul", "manag", "coordin"},
	"contain":     {"docker", "pod", "servic"},
	"kubernet":    {"k8s", "orchestr", "cluster"},
	"k8s":         {"kubernet", "orchestr", "contain"},

	// Storage / databases
	"databas":  {"store", "persist", "data"},
	"relat":    {"sql", "tabl", "schema"},
	"sql":      {"relat", "queri", "databas"},
	"nosql":    {"document", "key", "store"},
	"index":    {"search", "lookup", "catalog"},
	"cach":     {"memori", "store", "fast"},
	"persist":  {"store", "disk", "durabl"},
	"transact": {"acid", "commit", "rollback"},
	"complet":  {"consist", "durabil", "transact"},
	"replic":   {"backup", "mirror", "distribut"},
	"wal":      {"log", "write", "recov"},
	"lsm":      {"tree", "storag", "compact"},
	"sstabl":   {"immut", "disk", "storag"},
	"memtabl":  {"buffer", "memori", "write"},
	"bloom":    {"filter", "probablist", "lookup"},
	"compact":  {"merg", "clean", "lsm"},

	// Search
	"search": {"retriev", "queri", "find", "lookup"},
	"queri":  {"search", "request", "ask"},
	"rank":   {"sort", "order", "score"},
	"relev":  {"rank", "score", "match"},
	"semant": {"mean", "context", "embed"},
	"lexic":  {"keyword", "token", "term"},
	"fuzzi":    {"approxim", "edit", "typo"},
	"similar":  {"close", "near", "relat"},
	"document": {"file", "text", "record"},

	// Cloud / DevOps
	"cloud":   {"aw", "gcp", "azur", "infra"},
	"deploy":  {"ship", "releas", "launch"},
	"monitor": {"observ", "metric", "alert"},
	"observ":  {"monitor", "trace", "metric"},
	"metric":  {"monitor", "kpi", "measur"},
	"log":     {"event", "record", "trace"},
	"trace":   {"span", "debug", "log"},
	"alert":   {"notif", "warn", "alarm"},
	"ci":      {"build", "test", "automat"},
	"cd":      {"deploy", "releas", "deliv"},

	// Networking
	"network":  {"connect", "protocol", "packet"},
	"protocol": {"tcp", "http", "grpc"},
	"grpc":     {"rpc", "protobuf", "api"},
	"http":     {"rest", "api", "web"},
	"api":      {"endpoint", "interfac", "servic"},
	"request":  {"call", "queri", "invoc"},
	"respons":  {"repli", "result", "output"},

	// General programming
	"error":     {"except", "fault", "fail"},
	"except":    {"error", "panic", "fault"},
	"concurr":   {"parallel", "thread", "async"},
	"async":     {"concurr", "nonblock", "goroutin"},
	"goroutin":  {"thread", "concurr", "async"},
	"channel":   {"pipe", "stream", "messag"},
	"interfac":  {"contract", "api", "abstract"},
	"struct":    {"object", "type", "record"},
	"function":  {"method", "proc", "routin"},
	"algorithm": {"method", "approach", "logic"},
	"complex":   {"big", "notati", "perform"},
	"perform":   {"speed", "fast", "optim"},
	"memori":    {"heap", "alloc", "ram"},
	"cpu":       {"processor", "comput", "core"},
}

// Synonyms returns the synonym list for a stemmed term.
// Returns nil if no synonyms are registered.
// Thread-safe.
func Synonyms(stemmedTerm string) []string {
	synMu.RLock()
	defer synMu.RUnlock()
	return synonymMap[stemmedTerm]
}

// ExpandWithSynonyms takes a slice of stemmed query tokens and returns
// the original tokens plus any synonyms, deduplicated.
// Used exclusively at query time — never during indexing.
func ExpandWithSynonyms(tokens []string) []string {
	synMu.RLock()
	defer synMu.RUnlock()
	seen := make(map[string]struct{}, len(tokens)*2)
	result := make([]string, 0, len(tokens)*2)

	for _, tok := range tokens {
		if _, exists := seen[tok]; !exists {
			seen[tok] = struct{}{}
			result = append(result, tok)
		}
		for _, syn := range synonymMap[tok] {
			if _, exists := seen[syn]; !exists {
				seen[syn] = struct{}{}
				result = append(result, syn)
			}
		}
	}
	return result
}

var synMu sync.RWMutex

var synArrow = regexp.MustCompile(`^(.+?)\s*(<->|↔|->|→)\s*(.+)$`)

// commentIndex returns the byte index of a '#' that starts a trailing
// comment: at the start of the line, or preceded by whitespace. A '#' that
// is part of a token (e.g. the synonym "c#") is not treated as a comment
// marker, unlike a naive strings.Index(text, "#").
func commentIndex(s string) int {
	for i, r := range s {
		if r != '#' {
			continue
		}
		if i == 0 {
			return i
		}
		if prev := s[i-1]; prev == ' ' || prev == '\t' {
			return i
		}
	}
	return -1
}

// LoadSynonyms merges a synonyms file into the query-time synonym table. Each side is analysed
// with a (stemmed / Indic-folded) and must yield exactly one term. Call it before serving queries;
// it is safe but pointless to call concurrently with searches.
func LoadSynonyms(r io.Reader, a *StandardAnalyzer) (int, error) {
	sc := bufio.NewScanner(r)
	added, line := 0, 0
	synMu.Lock()
	defer synMu.Unlock()
	add := func(from, to string) {
		if slices.Contains(synonymMap[from], to) {
			return
		}
		synonymMap[from] = append(synonymMap[from], to)
		added++
	}
	for sc.Scan() {
		line++
		text := strings.TrimSpace(sc.Text())
		if i := commentIndex(text); i >= 0 {
			text = strings.TrimSpace(text[:i])
		}
		if text == "" {
			continue
		}
		m := synArrow.FindStringSubmatch(text)
		if m == nil {
			return added, fmt.Errorf("synonyms line %d: want 'a -> b' or 'a <-> b'", line)
		}
		l, rt := a.TokenizeExact(m[1]), a.TokenizeExact(m[3])
		if len(l) != 1 || len(rt) != 1 {
			return added, fmt.Errorf("synonyms line %d: each side must be exactly one non-stop-word term", line)
		}
		add(l[0], rt[0])
		if m[2] == "<->" || m[2] == "↔" {
			add(rt[0], l[0])
		}
	}
	return added, sc.Err()
}
