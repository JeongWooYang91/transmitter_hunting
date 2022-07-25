#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <GL/glut.h>
#include <GL/glu.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "toolfunctions.h"

#include <time.h>

#define N 172032 //131072
#define M 672
#define THRESHOLD 100000
#define GSIZE 1000
#define PNTSIZE 300
#define AUTO_END 10
#define ORTHO 80000
#define RADIUS 1500
#define RADSCALE 1000000000
#define LINE_SIZE 0.3

#define d2r(deg) (deg * PI / 180.0)
#define kill(s) (s->dead = true)

#define PI 3.14159265358979323846

#define MAPX 1273623389 
#define MAPY 363706170

#define NNUM 949
#define BNUM 117
#define FNUM 4

int nnum, bnum, fnum;
int point_mode = 0;

int width = 800, height = 800;
signal sig[N];
int selection_mode = 0; //generator: 0, detector: 1

clock_t total_testing_time = 0;

node *Nodes;
polygon *Buildings;
polygon *Forests;

line ga = {
	ga.x1 = -21800, ga.y1 = 13200, ga.x2 = -7000, ga.y2 = -8000
};

int toggle[10];

void load_file();
void clean_up();
void initialize();

__global__ void signal_calculation(signal *signal_list,
	const node *node_list, const polygon *building_list, const polygon *forest_list, const line *gtoa) {
	my_t gx = gtoa->x1;
	my_t gy = gtoa->y1;
	my_t ax = gtoa->x2;
	my_t ay = gtoa->y2;
	my_t zx, zy;
	int i = threadIdx.x + (blockIdx.x * blockDim.x);

	my_t px, py, test, tdist = 0, kdist = 0;
	signal sigref, sigblk;
	bool possible;

	signal *si = &signal_list[i];

	int autoend = -1;
	while (!si->dead && ++autoend < AUTO_END) {
		si->d = si->vy*si->x - si->vx*si->y;

		// case of detection
		possible = false;
		my_t d = (-si->vy*ax + si->vx*ay + si->d) / RADSCALE;
		if (-RADIUS <= d && d <= RADIUS) {
			if (si->vx != 0) {
				px = ax + (d*si->vy / RADSCALE);
				test = (px - si->x) ^ si->vx;
			}
			else {
				py = ay - (d*si->vx / RADSCALE);
				test = (py - si->y) ^ si->vy;
			}

			if (test > 0) {
				possible = true;
				zx = (si->x - ax);
				zy = (si->y - ay);
				tdist = zx*zx + zy*zy;
			}
		}


		// reflection test
		int n1, n2;
		int j, k;
		my_t test, kdist;
		my_t lx1, ly1, lx2, ly2;
		my_t Tnx, Tny, Td, pr;

		sigref.dead = true;
		int eid;

		for (j = 0; j < BNUM; j++) {
			// calculate reflection
			const polygon *p = &building_list[j];

			d = ((-si->vy)*p->x + (si->vx)*p->y + si->d) / RADSCALE;
			pr = p->radius;
			//possibly blocked if...
			if (-pr <= d && d <= pr)
			{
				for (k = 0; k < p->isize - 1; k++)
				{
					eid = 100 * i + k;
					if (si->eid == eid) continue;
					n1 = p->inodes[k];
					n2 = p->inodes[k + 1];

					lx1 = node_list[n1].x;
					ly1 = node_list[n1].y;
					lx2 = node_list[n2].x;
					ly2 = node_list[n2].y;

					Tnx = -si->vy;
					Tny = si->vx;
					Td = -(-si->vy*si->x + si->vx*si->y);
					my_t tb = Tnx*(lx2 - lx1) + Tny*(ly2 - ly1);

					if (tb == 0) { // parallel
						continue;
					}

					my_t t = -(Tnx*lx1 + Tny*ly1 + Td);
					if (t == 0 || t == tb) {
						continue;
					}
					if ((0 < t && t < tb) || (tb < t && t < 0)) {
						my_t px = lx1 + t*(lx2 - lx1) / tb;
						my_t py = ly1 + t*(ly2 - ly1) / tb;

						if (si->vx != 0) {
							test = (px - si->x) ^ si->vx;
						}
						else {
							test = (py - si->y) ^ si->vy;
						}

						if (test > 0) {
							zx = (si->x - px);
							zy = (si->y - py);
							kdist = zx*zx + zy*zy;
							if (kdist < 10) continue;
							if (sigref.dead || sigref.ss > kdist) { //if marked as alive
								my_t lnx = -(ly2 - ly1);
								my_t lny = (lx2 - lx1);
								my_t nv = lnx*si->vx + lny*si->vy;
								sigref.x = px;
								sigref.y = py;
								sigref.vx = si->vx - 2 * nv * lnx / (lnx*lnx + lny*lny);
								sigref.vy = si->vy - 2 * nv * lny / (lnx*lnx + lny*lny);
								sigref.ss = kdist;
								sigref.eid = eid;
								sigref.dead = false;
							}
						}
					}
				}
			}
		}

		// blocking test
		sigblk.dead = false;
		for (i = 0; i < FNUM; i++) {
			// calculate reflection
			const polygon *p = &forest_list[i];
			d = ((-si->vy)*p->x + (si->vx)*p->y + si->d) / RADSCALE;
			pr = p->radius;
			//possibly blocked if...
			if (-pr <= d && d <= pr)
			{
				for (k = 0; k < p->isize - 1; k++)
				{
					n1 = p->inodes[k];
					n2 = p->inodes[k + 1];

					lx1 = node_list[n1].x;
					ly1 = node_list[n1].y;
					lx2 = node_list[n2].x;
					ly2 = node_list[n2].y;

					Tnx = -si->vy;
					Tny = si->vx;
					Td = -(-si->vy*si->x + si->vx*si->y);//sigin->d;
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

						if (si->vx != 0) {
							test = (px - si->x) ^ si->vx;
						}
						else {
							test = (py - si->y) ^ si->vy;
						}

						if (test > 0) {
							zx = (si->x - px);
							zy = (si->y - py);
							kdist = zx*zx + zy*zy;
							if (!sigblk.dead || sigblk.ss > kdist) { //if marked as alive
																	 //printf("kdist = %lld\n", kdist);
								sigblk.x = px;
								sigblk.y = py;
								sigblk.ss = kdist;
								sigblk.dead = true;
							}
						}
					}
				}
			}
		}

		/*
		if (sigblk.dead) {
		if (possible && tdist < sigblk.ss) {
		printf("possible!\n");
		printf("tdist = %lld ", tdist);
		printf("sigblk.ss = %lld\n", sigblk.ss);
		si->ss += sqrt((float)tdist);
		break;
		}
		kill(si);
		break;
		}
		if (possible) {
		si->ss += sqrt((float)tdist);
		break;
		}
		kill(si);
		break;
		*/
		if (!sigref.dead) {
			if (sigblk.dead) {
				if (possible && tdist < sigref.ss && tdist < sigblk.ss) {
					si->ss += sqrt((float)tdist);
					break;
				}
				if (sigref.ss < sigblk.ss) {
					sigref.ss = sqrt(float(sigref.ss));
					sigref.ss += si->ss;
					*si = sigref;
					continue;
				}
				else {
					kill(si);
					break;
				}
			}
			else {
				if (possible && tdist < sigref.ss) {
					si->ss += sqrt((float)tdist);
					break;
				}
				else {
					sigref.ss = sqrt(float(sigref.ss));
					sigref.ss += si->ss;
					*si = sigref;
					continue;
				}
			}
		}
		else {
			if (sigblk.dead) {
				if (possible && tdist < sigblk.ss) {
					si->ss += sqrt((float)tdist);
					break;
				}
				else {
					kill(si);
					break;
				}
			}
		}

		if (possible)
			si->ss += sqrt((float)tdist);
		else
			kill(si);
		break;
	}
	if (autoend == AUTO_END) {
		kill(si);
	}
}

////////////////// cuda time
signal *dev_signals;
node *dev_nodes;
polygon *dev_buildings, *dev_forests;
line *dev_gtoa;

void freeCudaMemory() {
	cudaFree(dev_signals);
	cudaFree(dev_nodes);
	cudaFree(dev_buildings);
	cudaFree(dev_forests);
	cudaFree(dev_gtoa);
}

cudaError_t allocateCudaMemory() {
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	// Allocate GPU buffers for three vectors (two input, one output).
	cudaStatus = cudaMalloc((void**)&dev_gtoa, sizeof(line));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_signals, N * sizeof(signal));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_nodes, NNUM * sizeof(node));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.	
	cudaStatus = cudaMemcpy(dev_nodes, Nodes, NNUM * sizeof(node), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_buildings, BNUM * sizeof(polygon));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMemcpy(dev_buildings, Buildings, BNUM * sizeof(polygon), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_forests, FNUM * sizeof(polygon));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_forests, Forests, FNUM * sizeof(polygon), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


Error:
	return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t signalCalcWithCuda()
{
	clock_t tic = clock();
	cudaError_t cudaStatus;

	long double r;

	for (int i = 0; i < N; i++) {
		signal *si = &sig[i];
		r = d2r(360.0 * i / (long double)N);
		si->x = ga.x1;
		si->y = ga.y1;
		si->vx = cosl(r) * RADSCALE;
		si->vy = sinl(r) * RADSCALE;
		si->ss = 0;
		si->dead = false;
		si->eid = -1;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_signals, &sig, N * sizeof(signal), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_gtoa, &ga, sizeof(line), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	signal_calculation << <M, N / M >> >(dev_signals, dev_nodes, dev_buildings, dev_forests, dev_gtoa);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "signal_calculation launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching signal_calculation!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(sig, dev_signals, N * sizeof(signal), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	clock_t toc = clock();

	printf("elapsed: %d ms.\n", (toc - tic));
	total_testing_time += toc - tic;
	return cudaStatus;
}

void display()
{
	int gx = ga.x1;
	int gy = ga.y1;
	int ax = ga.x2;
	int ay = ga.y2;
	int i, j, in1, in2;
	cudaError_t cudaStatus;

	cudaStatus = signalCalcWithCuda();

	glClearColor(1, 1, 1, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glOrtho(-ORTHO, ORTHO, -ORTHO, ORTHO, -ORTHO, ORTHO);

	//

	glColor3d(0.9, 0.7, 0.9);

	//


	glColor3d(0, 1, 0);
	for (j = 0; j < FNUM; j++) {
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
	for (j = 0; j < BNUM; j++) {
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

}

void onMouseButton(int button, int state, int x, int y) {
	y = height - y - 1;
	//printf("mouse click on (%d, %d), (%I64d, %I64d)\n", x, y, dx, dy);

	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) {
			my_t dx = 2 * (x - width*0.5) / width * ORTHO;
			my_t dy = 2 * (y - height*0.5) / height * ORTHO;
			ga.x1 = dx;
			ga.y1 = dy;
			glutPostRedisplay();
		}
	}
	else if (button == GLUT_RIGHT_BUTTON) {
		if (state == GLUT_DOWN) {
			my_t dx = 2 * (x - width*0.5) / width * ORTHO;
			my_t dy = 2 * (y - height*0.5) / height * ORTHO;
			ga.x2 = dx;
			ga.y2 = dy;
			glutPostRedisplay();
		}
	}

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
			ga.x1 += GSIZE;
		}
		else {
			ga.x2 += GSIZE;
		}
	}
	if ((key == 'a')) { //right
		if (selection_mode == 0) {
			ga.x1 -= GSIZE;
		}
		else {
			ga.x2 -= GSIZE;
		}
	}
	if ((key == 'w')) { //up
		if (selection_mode == 0) {
			ga.y1 += GSIZE;
		}
		else {
			ga.y2 += GSIZE;
		}
	}
	if ((key == 's')) { //down
		if (selection_mode == 0) {
			ga.y1 -= GSIZE;
		}
		else {
			ga.y2 -= GSIZE;
		}
	}

	glutPostRedisplay();
}



void printga(int testnum) {
	printf("* Test %d (%lld, %lld, %lld, %lld) : ", testnum, MAPX + ga.x1, MAPY + ga.y1, MAPX + ga.x2, MAPY + ga.y2);
}

int main(int argc, char* argv[])
{
	initialize();

	printf("Number of signals: %d\n", N);
	/*
	ga = { 4200, 21200, 24800, 31800 };
	printga(1);
	signalCalcWithCuda();
	ga = { -37800, -39600, -51800, -29400 };
	printga(2);
	signalCalcWithCuda();
	ga = { -19000, 16600, -14400, 31200 };
	printga(3);
	signalCalcWithCuda();
	ga = { 20600, 31600, 11400, 41400 };
	printga(4);
	signalCalcWithCuda();
	ga = { 2600, 11000, 6600, 23200 };
	printga(5);
	signalCalcWithCuda();
	ga = { 41000, -1000, 44600, 11600 };
	printga(6);
	signalCalcWithCuda();
	ga = { 20200, 22200, 6200, 19800 };
	printga(7);
	signalCalcWithCuda();
	ga = { -13800, 18000, -6600, 27000 };
	printga(8);
	signalCalcWithCuda();
	ga = { 11400, 34600, -6600, 27000 };
	printga(9);
	signalCalcWithCuda();
	ga = { 11400, 34600, -13200, 37200 };
	printga(10);
	signalCalcWithCuda();
	printf("* Total testing time: %d ms, average time: %d ms.\n", total_testing_time, total_testing_time / 10);
	*/
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

void initialize() {
	load_file();
	allocateCudaMemory();
}

void load_file() {
	int count;
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

	int i, ni, maxr;

	fopen_s(&fp, "MapData.txt", "rt");
	if (fp != NULL)
	{
		nidx = bidx = fidx = 0;
		fscanf_s(fp, "i\t%d\t%d\t%d\n", &nnum, &bnum, &fnum);
		//deprecated dynamic allocation
		//printf("%d, %d, %d\n", nnum, bnum, fnum);
		Nodes = (node*)malloc(sizeof(node)*NNUM);
		Buildings = (polygon*)malloc(sizeof(polygon)*BNUM);
		Forests = (polygon*)malloc(sizeof(polygon)*FNUM);

		while (!feof(fp))
		{
			pstr = fgets(stmp, sizeof(stmp), fp);
			if (pstr == NULL) break;
			if (pstr[0] == 'n') {
				double lat, lon;
				sscanf_s(pstr, "n\t%lf\t%lf", &lat, &lon);
				Nodes[nidx].x = (my_t)(lon*1e7 - MAPX);
				Nodes[nidx].y = (my_t)(lat*1e7 - MAPY);
				nidx++;
			}
			if (pstr[0] == 'b') {
				count = 0; //except name tag
				for (char *c = pstr; *c != NULL; c++) {
					if (*c == '\t') count++;
				}

				//Buildings[bidx].inodes = (int*)malloc(sizeof(int)*count);
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

				/*
				Buildings[bidx].radius = 0;
				for (i = 0; i < Buildings[bidx].isize; i++) {
				ni = Buildings[bidx].inodes[i];
				maxr = (int)sqrt((Nodes[ni].x - Buildings[bidx].x)*(Nodes[ni].x - Buildings[bidx].x)
				+ (Nodes[ni].y - Buildings[fidx].y)*(Nodes[ni].y - Buildings[bidx].x)) + 1;
				if (Buildings[bidx].radius < maxr) {
				Buildings[bidx].radius = maxr;
				}
				}
				*/

				bidx++;
			}
			if (pstr[0] == 'f') {
				count = 0;
				for (char *c = pstr; *c != NULL; c++) {
					if (*c == '\t') count++;
				}

				//Forests[fidx].inodes = (int*)malloc(sizeof(int)*count);
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

				/*
				Forests[fidx].radius = 0;
				for (i = 0; i < Forests[fidx].isize; i++) {
				ni = Forests[fidx].inodes[i];
				maxr = (int)sqrt((Nodes[ni].x - Forests[fidx].x)*(Nodes[ni].x - Forests[fidx].x)
				+ (Nodes[ni].y - Forests[fidx].y)*(Nodes[ni].y - Forests[fidx].x)) + 1;
				if (Forests[fidx].radius < maxr) {
				Forests[fidx].radius = maxr;
				}
				}
				*/

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

void clean_up() {
	int i;
	free(Nodes);
	for (i = 0; i < BNUM; i++) {
		if (Buildings[i].inodes != NULL)
			free(Buildings[i].inodes);
	}
	free(Buildings);
	for (i = 0; i < FNUM; i++) {
		if (Forests[i].inodes != NULL)
			free(Forests[i].inodes);
	}
	free(Forests);

	freeCudaMemory();
}