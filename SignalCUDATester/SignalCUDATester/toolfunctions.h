#ifndef TOOL_FUNC_H_
#define TOOL_FUNC_H_

typedef __int64 my_t;

typedef struct {
	my_t x;
	my_t y;
	my_t vx;
	my_t vy;
	my_t d;
	my_t ss;
	int eid;
	bool dead;
} signal;

typedef struct {
	my_t x1;
	my_t y1;
	my_t x2;
	my_t y2;
} line;

typedef struct {
	my_t x;
	my_t y;
} node;

typedef struct {
	int inodes[60];
	int isize;
	my_t x, y, radius;
} polygon;

#endif
