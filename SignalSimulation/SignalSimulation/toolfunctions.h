#ifndef TOOL_FUNC_H_
#define TOOL_FUNC_H_

typedef __int64 my_t;

double dist(double x1, double y1, double x2, double y2);
my_t dist(my_t x1, my_t y1, my_t x2, my_t y2);
double veclen(my_t vx, my_t vy);
void normalize(double &vx, double &vy);
void normalize(long double &vx, long double &vy);
bool compare(__int64 i, __int64 j);
bool fzero(double d);
bool fzero(long double d);

#endif