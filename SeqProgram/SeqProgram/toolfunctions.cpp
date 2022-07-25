#include "toolfunctions.h";
#include <math.h>

double dist(double x1, double y1, double x2, double y2) {
	double dx = x2 - x1;
	double dy = y2 - y1;
	if (dx == 0.0) {
		if (dy < 0) dy = -dy;
		return dy;
	}
	if (dy == 0.0) {
		if (dx < 0) dx = -dx;
		return dx;
	}
	return sqrt(dx*dx + dy*dy);
}

my_t dist(my_t x1, my_t y1, my_t x2, my_t y2) {
	my_t dx = x2 - x1;
	my_t dy = y2 - y1;
	if (dx == 0) {
		return abs(dy);
	}
	if (dy == 0) {
		return abs(dx);
	}
	return (my_t)sqrt(dx*dx + dy*dy);
}

double veclen(my_t vx, my_t vy) {
	return (my_t)sqrt(vx*vx + vy*vy);
}

void normalize(double &vx, double &vy) {
	double len = sqrt(vx*vx + vy*vy);
	vx /= len;
	vy /= len;
}

void normalize(long double &vx, long double &vy) {
	long double len = sqrt(vx*vx + vy*vy);
	vx /= len;
	vy /= len;
}

bool compare(__int64 i, __int64 j) {
	return i < j;
}

bool fzero(double d) {
	return fabs(d) < 0.0000000000001;
}

bool fzero(long double d) {
	return fabsl(d) < 1e-128;
}