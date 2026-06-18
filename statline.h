// Human-readable statistics line for -s / --statistics output.
// Included by main.c (CPU) and worker_cuda.cu (GPU drain thread).
// Callers must include types.h and filters.h before this header.
#ifndef STATLINE_H
#define STATLINE_H

#include <stdio.h>

static void _sl_fmt_time(char *buf, size_t sz, double sec)
{
	long s;
	if (sec < 0.0 || sec != sec) { snprintf(buf, sz, "?"); return; }
	s = (long)sec;
	if      (s < 60)    snprintf(buf, sz, "%lds",       s);
	else if (s < 3600)  snprintf(buf, sz, "%ldm%02lds", s/60, s%60);
	else if (s < 86400) snprintf(buf, sz, "%ldh%02ldm", s/3600, (s%3600)/60);
	else                snprintf(buf, sz, "%ldd%02ldh", s/86400, (s%86400)/3600);
}

static void _sl_fmt_rate(char *buf, size_t sz, double r)
{
	if      (r >= 1e9) snprintf(buf, sz, "%.2fG/s", r/1e9);
	else if (r >= 1e6) snprintf(buf, sz, "%.1fM/s", r/1e6);
	else if (r >= 1e3) snprintf(buf, sz, "%.1fK/s", r/1e3);
	else               snprintf(buf, sz, "%.0f/s",  r);
}

static void _sl_fmt_count(char *buf, size_t sz, double n)
{
	if      (n >= 1e15) snprintf(buf, sz, "%.1fP", n/1e15);
	else if (n >= 1e12) snprintf(buf, sz, "%.1fT", n/1e12);
	else if (n >= 1e9)  snprintf(buf, sz, "%.1fB", n/1e9);
	else if (n >= 1e6)  snprintf(buf, sz, "%.1fM", n/1e6);
	else if (n >= 1e3)  snprintf(buf, sz, "%.1fK", n/1e3);
	else                snprintf(buf, sz, "%.0f",  n);
}

// Expected candidates per match: 2^popcount(ifiltermask) / nfilters.
// Returns 0 if the filter type doesn't support bit-counting (PCRE2, etc.).
static double _sl_pow2(int n)
{
	double r = 1.0;
	while (n-- > 0) r *= 2.0;
	return r;
}

#ifdef INTFILTER
static double _sl_difficulty(void)
{
	const unsigned char *p = (const unsigned char *)&ifiltermask;
	size_t i;
	int nbits = 0;
	for (i = 0; i < sizeof(ifiltermask); i++)
		nbits += __builtin_popcount((unsigned)p[i]);
	size_t n = filters_count();
	return (n > 0 && nbits > 0) ? _sl_pow2(nbits) / (double)n : 0.0;
}
#elif defined(BINFILTER)
static double _sl_difficulty(void)
{
	size_t i, n = filters_count();
	double prob_sum = 0.0;
	if (n == 0) return 0.0;
	for (i = 0; i < n; i++) {
		struct binfilter *bf = &VEC_BUF(filters, i);
		int nbits = (int)(bf->len) * 8 + __builtin_popcount((unsigned)bf->mask);
		prob_sum += 1.0 / _sl_pow2(nbits);
	}
	return prob_sum > 0.0 ? 1.0 / prob_sum : 0.0;
}
#else
static double _sl_difficulty(void) { return 0.0; }
#endif

// Print one human-readable stats line to fp.
// elapsed_us: microseconds since start; total_found: keys written so far.
static void print_stats_line(FILE *fp, double calcpersec,
                              u64 elapsed_us, u64 total_found)
{
	char tbuf[24], rbuf[24], dbuf[24], e50[24], e90[24];
	double elapsed_s = (double)elapsed_us * 1e-6;
	double diff = _sl_difficulty();

	_sl_fmt_time(tbuf, sizeof(tbuf), elapsed_s);
	_sl_fmt_rate(rbuf, sizeof(rbuf), calcpersec);

	if (diff > 0.0 && calcpersec > 0.0) {
		_sl_fmt_count(dbuf, sizeof(dbuf), diff);
		_sl_fmt_time(e50, sizeof(e50), diff * 0.693147 / calcpersec);
		_sl_fmt_time(e90, sizeof(e90), diff * 2.302585 / calcpersec);
		fprintf(fp,
		    "> elapsed: %8s | speed: %9s | 1:%s | ETA 50%%: %s  90%%: %s | found: %llu\n",
		    tbuf, rbuf, dbuf, e50, e90, (unsigned long long)total_found);
	} else {
		fprintf(fp,
		    "> elapsed: %8s | speed: %9s | found: %llu\n",
		    tbuf, rbuf, (unsigned long long)total_found);
	}
}

#endif /* STATLINE_H */
