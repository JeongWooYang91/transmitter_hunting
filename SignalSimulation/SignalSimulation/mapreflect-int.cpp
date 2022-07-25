//#define _CRT_SECURE_NO_WARNINGS

#include <GL/glut.h>
#include <GL/glu.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "toolfunctions.h"

#include <time.h>

#define N 7200
#define THRESHOLD 100000
#define GSIZE 1000
#define PNTSIZE 300
#define LINE_SIZE 3
#define AUTO_END 20
#define kill(s) (s->dead = true)
#define ORTHO 80000
#define RADIUS PNTSIZE

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
	int *inodes, isize;
	my_t x, y, radius;
} polygon;


int width = 800, height = 800;
double PI = acos(-1.0);
signal sig[N];
int selection_mode = 0; //generator: 0, detector: 1
my_t gx = -21800, gy = 13200; //generation point
my_t ax = -7000, ay = -8000; //accepting point

my_t mapx = 1273623389;
my_t mapy = 363706170;

int nnum, bnum, fnum;

node *Nodes;
polygon *Buildings;
polygon *Forests;
line reflecting[10][50000];
int nreflect[10];
node meeting[50000];
int nmeeting = 0;
int point_mode = 0;
int toggle[10];

void signal_set(signal *s, my_t x, my_t y, my_t vx, my_t vy, my_t ss, bool dead) {
	s->x = x;
	s->y = y;
	s->vx = vx;
	s->vy = vy;
	s->ss = ss;
	s->dead = dead;
	s->eid = -1;
}

void record(signal *a, signal *b, int autoend) {
	line *tl = &reflecting[autoend][nreflect[autoend]++];
	tl->x1 = a->x;
	tl->y1 = a->y;
	tl->x2 = b->x;
	tl->y2 = b->y;
}

void signal_deepcpy(signal *a, signal *b) {
	b->x = a->x;
	b->y = a->y;
	b->vx = a->vx;
	b->vy = a->vy;
	b->ss = a->ss;
	b->dead = a->dead;
	b->eid = a->eid;
}

void prt(signal *s) {
	//printf("prt: p(%4.3f, %4.3f), v(%f, %f), s(%f)\n", s->x, s->y, s->vx, s->vy, s->ss);
	printf("prt: p(%I64d, %I64d), v(%I64d, %I64d), s(%I64d)\n", s->x, s->y, s->vx, s->vy, s->ss);
}

long double d2r(long double deg) {
	return deg * PI / 180.0;
}

void forest_block(signal *sigin, signal *sigout) {
	int n1, n2;
	int i, j, k;
	my_t test;
	my_t kdist;
	my_t lx1, ly1, lx2, ly2;
	my_t Tnx, Tny, Td;
	
	if (sigin->dead) {
		printf("should not be happened\n");
		sigout->dead = true;
		return;
	}
	sigout->dead = false;

	for (i = 0; i < fnum; i++) {
		// calculate reflection
		polygon *p = &Forests[i];

		my_t d = ((-sigin->vy)*p->x + (sigin->vx)*p->y + sigin->d);
		my_t vl = veclen(sigin->vx, sigin->vy);
		my_t pr = p->radius*vl;
		//possibly blocked if...
		if (-pr <= d && d <= pr)
		{
			for (k = 0; k < p->isize - 1; k++)
			{
				n1 = p->inodes[k];
				n2 = p->inodes[k + 1];

				lx1 = Nodes[n1].x;
				ly1 = Nodes[n1].y;
				lx2 = Nodes[n2].x;
				ly2 = Nodes[n2].y;

				Tnx = -sigin->vy;
				Tny = sigin->vx;
				Td = -(-sigin->vy*sigin->x + sigin->vx*sigin->y);//sigin->d;
				// p' = p1 + t(p2-p1), T(dot)p' = 0
				// t = -(T(dot)p1) / (T(dot)(p2 - p1))
				my_t tb = Tnx*(lx2 - lx1) + Tny*(ly2 - ly1);
				
				if (tb == 0) { // parallel
					continue;
				}
				
				my_t t = -(Tnx*lx1 + Tny*ly1 + Td);
				if (t == 0 || t == tb) continue;
				if ((0 < t && t < tb) || (tb < t && t < 0)) {
					my_t px = lx1 + t*(lx2 - lx1) / tb;
					my_t py = ly1 + t*(ly2 - ly1) / tb;

					if (sigin->vx != 0) {
						test = (px - sigin->x) ^ sigin->vx;
					}
					else {
						test = (py - sigin->y) ^ sigin->vy;
					}

					if (test > 0) {
						kdist = dist(sigin->x, sigin->y, px, py);
						if (!sigout->dead || sigout->ss > kdist) { //if marked as alive
							sigout->x = px;
							sigout->y = py;
							sigout->ss = kdist;
							sigout->dead = true;
						}
					}
				}
			}
		}
	}
}

void building_reflection(signal *sigin, signal *sigout) {
	int n1, n2;
	int i, j, k;
	my_t test;
	my_t kdist;
	my_t lx1, ly1, lx2, ly2;
	my_t Tnx, Tny, Td;

	if (sigin->dead) {
		printf("should not be happened\n");
		sigout->dead = true;
		return;
	}
	sigout->dead = true;
	int eid;

	for (i = 0; i < bnum; i++) {
		// calculate reflection
		polygon *p = &Buildings[i];

		my_t d = ((-sigin->vy)*p->x + (sigin->vx)*p->y + sigin->d);
		my_t vl = veclen(sigin->vx, sigin->vy);
		my_t pr = p->radius*vl;
		//possibly blocked if...
		if (-pr <= d && d <= pr)
		{
			for (k = 0; k < p->isize - 1; k++)
			{
				eid = 100 * i + k;
				if (sigin->eid == eid) continue;
				n1 = p->inodes[k];
				n2 = p->inodes[k + 1];

				lx1 = Nodes[n1].x;
				ly1 = Nodes[n1].y;
				lx2 = Nodes[n2].x;
				ly2 = Nodes[n2].y;

				Tnx = -sigin->vy;
				Tny = sigin->vx;
				Td = -(-sigin->vy*sigin->x + sigin->vx*sigin->y);//sigin->d;
																 // p' = p1 + t(p2-p1), T(dot)p' = 0
																 // t = -(T(dot)p1) / (T(dot)(p2 - p1))
				my_t tb = Tnx*(lx2 - lx1) + Tny*(ly2 - ly1);

				if (tb == 0) { // parallel
					continue;
				}

				my_t t = -(Tnx*lx1 + Tny*ly1 + Td);
				if (t == 0 || t == tb) {
					kill(sigout);
					return;
				}
				if ((0 < t && t < tb) || (tb < t && t < 0)) {
					my_t px = lx1 + t*(lx2 - lx1) / tb;
					my_t py = ly1 + t*(ly2 - ly1) / tb;

					if (sigin->vx != 0) {
						test = (px - sigin->x) ^ sigin->vx;
					}
					else {
						test = (py - sigin->y) ^ sigin->vy;
					}

					if (test > 0) {
						kdist = dist(sigin->x, sigin->y, px, py);
						if (kdist < 10) continue;
						if (sigout->dead || sigout->ss > kdist) { //if marked as alive
							my_t lnx = -(ly2 - ly1);
							my_t lny = (lx2 - lx1);
							my_t nv = lnx*sigin->vx + lny*sigin->vy;
							sigout->x = px;
							sigout->y = py;
							sigout->vx = sigin->vx - 2 * nv * lnx / (lnx*lnx + lny*lny);
							sigout->vy = sigin->vy - 2 * nv * lny / (lnx*lnx + lny*lny);
							sigout->ss = kdist;
							sigout->eid = eid;
							sigout->dead = false;
						}
					}
				}
			}
		}
	}
}

void signal_calculation() {
	nmeeting = 0;
	int i, j, k;
	for (i = 0; i < 10; i++) nreflect[i] = 0;

	my_t t, px, py, test, tdist = 0, kdist = 0;
	signal sigref, sigblk;
	bool possible;
	for (i = 0; i < N; i++) {
		long double r = d2r(360.0 * i / (long double)N);
		signal_set(&sig[i], gx, gy, cos(r)*100000000, sin(r)*100000000, 0, false);
		//signal_set(&sig[i], gx, gy, ax-gx, ay-gy, 0, false);
	}
	for (i = 0; i < N; i++) {
		signal *si = &sig[i];
		int autoend = -1; //unknown error--> need to be fixed
		while (!si->dead && ++autoend < 10) {
			//prt(si);
			si->d = si->vy*si->x - si->vx*si->y;

			// case of detection
			possible = false;
			my_t d = - si->vy*ax + si->vx*ay + si->d;
			my_t vl = veclen(si->vx, si->vy);
			my_t vq = si->vx*si->vx + si->vy*si->vy;
			my_t pr = RADIUS*vl;
			if (-pr <= d && d <= pr) {
				if (si->vx != 0) {
					px = ax + (d*si->vy / vq);
					test = (px - si->x) ^ si->vx;
				}
				else {
					py = ay - (d*si->vx / vq);
					test = (py - si->y) ^ si->vy;
				}

				if (test > 0) {
					possible = true;
					tdist = dist(si->x, si->y, ax, ay);
				}
			}

			
			// reflection test
			building_reflection(si, &sigref);
			// blocking test
			forest_block(si, &sigblk);

			if (!sigref.dead) {
				if (sigblk.dead) {
					if (possible && tdist < sigref.ss && tdist < sigblk.ss) {
						si->ss += tdist;
						break;
					}
					if (sigref.ss < sigblk.ss) {
						sigref.ss += si->ss;
						record(&sigref, si, autoend);
						signal_deepcpy(&sigref, si);
						continue;
					}
					else {
						kill(si);
						break;
					}
				}
				else {
					if (possible && tdist < sigref.ss) {
						si->ss += tdist;
						break;
					}
					else {
						sigref.ss += si->ss;
						record(&sigref, si, autoend);
						signal_deepcpy(&sigref, si);
						continue;
					}
				}
			}
			else {
				if (sigblk.dead) {
					if (possible && tdist < sigblk.ss) {
						si->ss += tdist;
						break;
					}
					else {
						kill(si);
						break;
					}
				}
			}

			if (possible)
				si->ss += tdist;
			else
				kill(si);
			break;
		}
		if (autoend == 10) {
			kill(si);
		}
	}
}

void load_file() {
	int i, count;
	FILE * fp;
	char stmp[255];
	char *pstr;
	char *context = NULL;
	char *token;
	const char del[3] = "\t\n";
	int nidx, bidx, fidx;

	bool firstline = true;
	bool isname = true;
	int ti;
	int tokidx;
	my_t mxx, mxy, mix, miy;

	fopen_s(&fp, "MapData.txt", "rt");
	if (fp != NULL)
	{
		nidx = bidx = fidx = 0;
		fscanf_s(fp, "i\t%d\t%d\t%d\n", &nnum, &bnum, &fnum);
		printf("%d, %d, %d\n", nnum, bnum, fnum);
		Nodes = (node*)malloc(sizeof(node)*nnum);
		Buildings = (polygon*)malloc(sizeof(polygon)*bnum);
		Forests = (polygon*)malloc(sizeof(polygon)*fnum);

		while (!feof(fp))
		{
			pstr = fgets(stmp, sizeof(stmp), fp);
			if (pstr == NULL) break;
			if (pstr[0] == 'n') {
				double lat, lon;
				sscanf_s(pstr, "n\t%lf\t%lf", &lat, &lon);
				Nodes[nidx].x = (my_t)(lon*1e7 - mapx);
				Nodes[nidx].y = (my_t)(lat*1e7 - mapy);
				nidx++;
			}
			if (pstr[0] == 'b') {
				count = 0; //except name tag
				for (char *c = pstr; *c != NULL; c++) {
					if (*c == '\t') count++;
				}

				Buildings[bidx].inodes = (int*)malloc(sizeof(int)*count);
				Buildings[bidx].isize = count;
				mxx = mxy = -99999;
				mix = miy = 99999;

				token = strtok_s(pstr, del, &context);
				tokidx = 0;
				isname = true;
				while (token != NULL)
				{
					if (isname) {
						token = strtok_s(NULL, del, &context);
						isname = false;
						continue;
					}
					sscanf_s(token, "%d", &ti);
					Buildings[bidx].inodes[tokidx] = ti;
					if (mxx < Nodes[ti].x)
						mxx = Nodes[ti].x;
					if (mxy < Nodes[ti].y)
						mxy = Nodes[ti].y;
					if (mix > Nodes[ti].x)
						mix = Nodes[ti].x;
					if (miy > Nodes[ti].y)
						miy = Nodes[ti].y;

					token = strtok_s(NULL, del, &context);
					tokidx++;
				}
				Buildings[bidx].x = (mxx + mix) / 2;
				Buildings[bidx].y = (mxy + miy) / 2;
				Buildings[bidx].radius = sqrt((mxx - mix)*(mxx - mix) + (mxy - miy)*(mxy - miy)) / 2;

				bidx++;
			}
			if (pstr[0] == 'f') {
				count = 0;
				for (char *c = pstr; *c != NULL; c++) {
					if (*c == '\t') count++;
				}

				Forests[fidx].inodes = (int*)malloc(sizeof(int)*count);
				Forests[fidx].isize = count;
				mxx = mxy = -99999;
				mix = miy = 99999;

				token = strtok_s(pstr, del, &context);
				tokidx = 0;
				isname = true;
				while (token != NULL)
				{
					if (isname) {
						token = strtok_s(NULL, del, &context);
						isname = false;
						continue;
					}
					sscanf_s(token, "%d", &ti);
					Forests[fidx].inodes[tokidx] = ti;
					if (mxx < Nodes[ti].x)
						mxx = Nodes[ti].x;
					if (mxy < Nodes[ti].y)
						mxy = Nodes[ti].y;
					if (mix > Nodes[ti].x)
						mix = Nodes[ti].x;
					if (miy > Nodes[ti].y)
						miy = Nodes[ti].y;

					token = strtok_s(NULL, del, &context);
					tokidx++;
				}
				Forests[fidx].x = (mxx + mix) / 2;
				Forests[fidx].y = (mxy + miy) / 2;
				Forests[fidx].radius = sqrt((mxx - mix)*(mxx - mix) + (mxy - miy)*(mxy - miy)) / 2;

				fidx++;
			}
		}
		fclose(fp);
	}
	else
	{
		//file not exist
	}
}

void initialize() {
	load_file();
}

void clean_up() {
	int i;
	free(Nodes);
	for (i = 0; i < bnum; i++) {
		if (Buildings[i].inodes != NULL)
			free(Buildings[i].inodes);
	}
	free(Buildings);
	for (i = 0; i < fnum; i++) {
		if (Forests[i].inodes != NULL)
			free(Forests[i].inodes);
	}
	free(Forests);
}

typedef struct {
	double r;
	double g;
	double b;
} color;

double min(double a, double b) {
	return a < b ? a : b;
}

double max(double a, double b) {
	return a > b ? a : b;
}

void spec(double val, color *c) {
	c->r = 0.6*val;
	c->g = 0.6*val;
	c->b = 0.8*val;
}

void display()
{

	clock_t tic = clock();

	int i, j, in1, in2;
	signal_calculation();

	clock_t toc = clock();

	printf("Calculation Elapsed: %d ms.\n", (toc - tic));
	tic = toc;

	glClearColor(1, 1, 1, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glOrtho(-ORTHO, ORTHO, -ORTHO, ORTHO, -ORTHO, ORTHO);

	//

	glColor3d(0.9, 0.7, 0.9);
	for (j = 0; j < 10; j++) {
		if (toggle[j]) {
			for (i = 0; i < nreflect[j]; i++) {
				line *tl = &reflecting[j][i];
				glBegin(GL_LINES);
				glVertex3f(tl->x1, tl->y1, 0.0f);
				glVertex3f(tl->x2, tl->y2, 0.0f);
				glEnd();
			}
		}
	}
	for (i = 0; i < nmeeting; i++) {
		node *n = &meeting[i];
		long double kx = n->x;
		long double ky = n->y;
		glColor3d(0, 0.5, 0.5);
		glBegin(GL_POLYGON);
		glVertex3f(kx + PNTSIZE, ky - PNTSIZE, 0.0f);
		glVertex3f(kx + PNTSIZE, ky + PNTSIZE, 0.0f);
		glVertex3f(kx - PNTSIZE, ky + PNTSIZE, 0.0f);
		glVertex3f(kx - PNTSIZE, ky - PNTSIZE, 0.0f);
		glEnd();
	}

	//


	glColor3d(0, 1, 0);
	for (j = 0; j < fnum; j++) {
		for (i = 0; i < Forests[j].isize - 1; i++) {
			in1 = Forests[j].inodes[i];
			in2 = Forests[j].inodes[i + 1];
			glBegin(GL_LINES);
			glVertex3f(Nodes[in1].x, Nodes[in1].y, 0.0f);
			glVertex3f(Nodes[in2].x, Nodes[in2].y, 0.0f);
			glEnd();
			if (point_mode) {
				glBegin(GL_POLYGON);
				glVertex3f(Nodes[in1].x + PNTSIZE, Nodes[in1].y - PNTSIZE, 0.0f);
				glVertex3f(Nodes[in1].x + PNTSIZE, Nodes[in1].y + PNTSIZE, 0.0f);
				glVertex3f(Nodes[in1].x - PNTSIZE, Nodes[in1].y + PNTSIZE, 0.0f);
				glVertex3f(Nodes[in1].x - PNTSIZE, Nodes[in1].y - PNTSIZE, 0.0f);
				glEnd();
			}
		}
	}

	glColor3d(0.5, 0.5, 0.5);
	for (j = 0; j < bnum; j++) {
		for (i = 0; i < Buildings[j].isize - 1; i++) {
			in1 = Buildings[j].inodes[i];
			in2 = Buildings[j].inodes[i + 1];
			glBegin(GL_LINES);
			glVertex3f(Nodes[in1].x, Nodes[in1].y, 0.0f);
			glVertex3f(Nodes[in2].x, Nodes[in2].y, 0.0f);
			glEnd();
			if (point_mode) {
				glBegin(GL_POLYGON);
				glVertex3f(Nodes[in1].x + PNTSIZE, Nodes[in1].y - PNTSIZE, 0.0f);
				glVertex3f(Nodes[in1].x + PNTSIZE, Nodes[in1].y + PNTSIZE, 0.0f);
				glVertex3f(Nodes[in1].x - PNTSIZE, Nodes[in1].y + PNTSIZE, 0.0f);
				glVertex3f(Nodes[in1].x - PNTSIZE, Nodes[in1].y - PNTSIZE, 0.0f);
				glEnd();
			}
		}
	}

	/*
	for (int i = 0; i < N; i++) {
		glBegin(GL_LINES);
		glVertex3f(sig[i].x, sig[i].y, 0.0f);
		glVertex3f(sig[i].x + sig[i].vx, sig[i].y + sig[i].vy, 0.0f);
		glEnd();

	}
	*/

	
	glColor3d(0, 0, 1);
	for (int i = 0; i < N; i++) {
		signal *si = &sig[i];
		if (!si->dead && si->ss > 0 && si->ss < THRESHOLD) {
			glBegin(GL_LINES);
			glVertex3f(ax, ay, 0.0f);
			double m = LINE_SIZE / (double)si->ss;
			my_t tx = m*(-si->vx);
			my_t ty = m*(-si->vy);
			glVertex3f(ax + tx, ay + ty, 0.0f);
			glEnd();
		}
	}


	


	glColor3d(0, 0, 0.5);
	glBegin(GL_POLYGON);
	glVertex3f(ax + GSIZE, ay - GSIZE, 0.0f);
	glVertex3f(ax + GSIZE, ay + GSIZE, 0.0f);
	glVertex3f(ax - GSIZE, ay + GSIZE, 0.0f);
	glVertex3f(ax - GSIZE, ay - GSIZE, 0.0f);
	glEnd();

	glColor3d(1, 0, 0);
	glBegin(GL_POLYGON);
	glVertex3f(gx + GSIZE, gy - GSIZE, 0.0f);
	glVertex3f(gx + GSIZE, gy + GSIZE, 0.0f);
	glVertex3f(gx - GSIZE, gy + GSIZE, 0.0f);
	glVertex3f(gx - GSIZE, gy - GSIZE, 0.0f);
	glEnd();

	glFlush();

	toc = clock();

	printf("Drawing Elapsed: %d ms.\n", (toc - tic));
}

void onMouseButton(int button, int state, int x, int y) {
	y = height - y - 1;
	my_t dx = 2 * (x - width*0.5) / width * ORTHO;
	my_t dy = 2 * (y - height*0.5) / height * ORTHO;
	//printf("mouse click on (%d, %d), (%I64d, %I64d)\n", x, y, dx, dy);

	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) {
			gx = dx;
			gy = dy;
		}
	}
	else if (button == GLUT_RIGHT_BUTTON) {
		if (state == GLUT_DOWN) {
			ax = dx;
			ay = dy;
		}
	}

	glutPostRedisplay();
}

void onMouseDrag(int x, int y) {
}

/*********************************************************************************
* Call this part whenever user types keyboard.
* This part is called in main() function by registering on glutKeyboardFunc(onKeyPress).
**********************************************************************************/
void onKeyPress(unsigned char key, int x, int y) {
	if ('0' <= key && key <= '9') {
		toggle[key - '0'] = !toggle[key - '0'];
		for (int i = 0; i < 10; i++) {
//			printf("%d ", toggle[i]);
		}
//		printf("\n");
	}
	if ((key == 'p')) { //receiver
		point_mode = !point_mode;
	}
	if ((key == 'g')) { //generator
		selection_mode = 0;
	}
	if ((key == 'r')) { //receiver
		selection_mode = 1;
	}
	if ((key == 'd')) { //left
		if (selection_mode == 0) {
			gx += GSIZE;
		}
		else {
			ax += GSIZE;
		}
	}
	if ((key == 'a')) { //right
		if (selection_mode == 0) {
			gx -= GSIZE;
		}
		else {
			ax -= GSIZE;
		}
	}
	if ((key == 'w')) { //up
		if (selection_mode == 0) {
			gy += GSIZE;
		}
		else {
			ay += GSIZE;
		}
	}
	if ((key == 's')) { //down
		if (selection_mode == 0) {
			gy -= GSIZE;
		}
		else {
			ay -= GSIZE;
		}
	}

	glutPostRedisplay();
}


int main(int argc, char* argv[])
{
	initialize();
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutCreateWindow("CS408 SIGNAL TEST");
	glutDisplayFunc(display);
	glutMouseFunc(onMouseButton);					// Register onMouseButton function to call that when user moves mouse.
	glutMotionFunc(onMouseDrag);					// Register onMouseDrag function to call that when user drags mouse.

	glutKeyboardFunc(onKeyPress);
	glutMainLoop();
	clean_up();

	return 0;
}
