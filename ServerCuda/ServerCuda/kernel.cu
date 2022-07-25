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
#define NCOMPASS 360

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
int num_hounds;

int width = 800, height = 800;
signal sig[N];
int compass[10][NCOMPASS];
int count[10][NCOMPASS];
int selection_mode = 0; //generator: 0, detector: 1

node *Nodes;
polygon *Buildings;
polygon *Forests;
info GameInfo;


////////////////// cuda time
signal *dev_signals;
node *dev_nodes;
polygon *dev_buildings, *dev_forests;
info *dev_info;

int toggle[10];

void load_file();
void clean_up();
void initialize();

__global__ void signal_calculation(signal *signal_list,
	const node *node_list, const polygon *building_list, const polygon *forest_list, const info *dev_info) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	my_t gx = dev_info->gx;
	my_t gy = dev_info->gy;
	my_t ax = dev_info->ax[i / (N / dev_info->num_hounds)];
	my_t ay = dev_info->ay[i / (N / dev_info->num_hounds)];
	//if(i % 10000 == 0)
	//	printf("gx = %lld, gy = %lld, ax = %lld, ay = %lld\n", gx, gy, ax, ay);
	my_t zx, zy;

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
							if (!sigblk.dead || sigblk.ss > kdist) {
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


void freeCudaMemory() {
	cudaFree(dev_signals);
	cudaFree(dev_nodes);
	cudaFree(dev_buildings);
	cudaFree(dev_forests);
	cudaFree(dev_info);
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
	cudaStatus = cudaMalloc((void**)&dev_info, sizeof(info));
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
	cudaError_t cudaStatus;

	long double r;

	int modN = N - (N%GameInfo.num_hounds);
	int eachN = (N / GameInfo.num_hounds);
	int i;
	for (i = 0; i < N; i++) {
		signal *si = &sig[i];
		if (i < modN) {
			r = d2r(360.0 * (i % eachN) / (long double)eachN);
			si->x = GameInfo.gx;
			si->y = GameInfo.gy;
			si->vx = cosl(r) * RADSCALE;
			si->vy = sinl(r) * RADSCALE;
			si->ss = 0;
			si->dead = false;
			si->eid = -1;
		}
		else {
			kill(si);
		}
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_signals, &sig, N * sizeof(signal), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_info, &GameInfo, sizeof(info), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	signal_calculation << <M, N / M >> >(dev_signals, dev_nodes, dev_buildings, dev_forests, dev_info);

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
	return cudaStatus;
}

void convertToCompass() {
	int i, j, hidx, sidx;
	double sum;

	for (i = 0; i < num_hounds; i++) {
		for (j = 0; j < NCOMPASS; j++) {
			compass[i][j] = 0;
			count[i][j] = 0; //initialzie
		}
	}

	int rN = N - (N % num_hounds);
	int eachN = rN / num_hounds;
	int deg;
	for (i = 0; i < rN; i++) {
		hidx = i / eachN;

		deg = (int)(atan2(-sig[i].vy, -sig[i].vx) * 180 / PI);
		if (deg < 0) deg += 360;
		sidx = NCOMPASS * deg / 360;

		if (!sig[i].dead) {
			compass[hidx][sidx] = 1000000000 / sig[i].ss;
			count[hidx][sidx] = 1;
		}
		//compass[hidx]
	}
	for (i = 0; i < hidx; i++) {
		for (j = 0; j < NCOMPASS; j++) {
			if(count[i][j] != 0)
				compass[i][j] /= count[i][j];
		}
	}
}

int main(int argc, char* argv[])
{
	if (argc == 1) return 0;
	initialize();

	int i, j;
	my_t lat, lon;
	lon = atoll(argv[1]);
	lat = atoll(argv[2]);
	
	GameInfo.gx = lon - (my_t)MAPX;
	GameInfo.gy = lat - (my_t)MAPY;

	sscanf_s(argv[3], "%d", &num_hounds);
	GameInfo.num_hounds = num_hounds;
	for (i = 0; i < num_hounds; i++) {
		lon = atoll(argv[4 + i*2]);
		lat = atoll(argv[5 + i*2]);
		GameInfo.ax[i] = lon - (my_t)MAPX;
		GameInfo.ay[i] = lat - (my_t)MAPY;
	}

	signalCalcWithCuda();
	
	convertToCompass();


	fprintf(stdout, "[");
	for (j = 0; j < num_hounds; j++) {

		fprintf(stdout, "[");
		for (i = 0; i < NCOMPASS; i++) {
			if (i == NCOMPASS - 1) {
				fprintf(stdout, "%d", compass[j][i]);
			}
			else {
				fprintf(stdout, "%d, ", compass[j][i]);
			}
		}

		if (j < num_hounds - 1) {
			fprintf(stdout, "], ");
		}
		else fprintf(stdout, "]");
	}

	fprintf(stdout, "]");

	/*
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutCreateWindow("CS408 SIGNAL TEST");
	glutDisplayFunc(display);
	glutMouseFunc(onMouseButton);					// Register onMouseButton function to call that when user moves mouse.
	glutMotionFunc(onMouseDrag);					// Register onMouseDrag function to call that when user drags mouse.

	glutKeyboardFunc(onKeyPress);
	glutMainLoop();

	*/
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
				mxx = mxy = -999999999;
				mix = miy = 999999999;

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
				mxx = mxy = -999999999;
				mix = miy = 999999999;

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
		fprintf(stderr, "File not found!\n");
	}
}

void clean_up() {
	int i;
	free(Nodes);
	/*
	for (i = 0; i < BNUM; i++) {
		if (Buildings[i].inodes != NULL)
			free(Buildings[i].inodes);
	}
	*/
	free(Buildings);
	/*
	for (i = 0; i < FNUM; i++) {
		if (Forests[i].inodes != NULL)
			free(Forests[i].inodes);
	}
	*/
	free(Forests);

	freeCudaMemory();
}