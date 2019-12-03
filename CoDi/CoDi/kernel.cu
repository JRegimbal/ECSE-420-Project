
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "codi_cell.h"
#include "chromosome.h"
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <chrono>

#ifndef __CUDACC_RTC__
#define __CUDACC_RTC__
#endif // !(__CUDACC_RTC__)

#define MAX(a, b) (a > b ? a : b)

using namespace std;

const float neuronSeedP = 0.10;
const unsigned int sig = 100;
const int threads = 20;

/*
We want x and y to be closer in memory than z because of locality.
Access grid (i, j, k) at index (i + x * j + x * y * k)
In a grid of dimensions x by y by z
*/

__device__ bool lock(unsigned int* loc) {
	int old = atomicCAS(loc, 0, 1);
	if (old == 0 && *loc == 1) {
		return true;
	}
	return false;
}

__device__ void unlock(unsigned int* loc) {
	if (*loc == 1) {
		atomicCAS(loc, 1, 0);
	}
}

/*
We need to use barriers for some control and semaphores to avoid write conflicts
Currently this assumes each x-y slice has exactly one block assigned to it.
Step 1: determine which cells each thread is responsible for
Step 2: if the cell is initialized and hasn't grown, follow growth for it. otherwise continue to next cell.
Step 3: take ownership of cell(s) to grow into using the locks
Step 4: grow
Step 5: mark this cell as having grown and move on to next cell
*/
__global__ void growKernel(Cell* grid, unsigned int x, unsigned int y, unsigned int z, unsigned int iterations) {
	//extern __shared__  unsigned int sharedMemory[];
	__shared__ unsigned int locks[1000];
	// We need to initialize locks to zero
	for (int i = threadIdx.x; i < x * y; i += blockDim.x) {
		locks[i] = 0;
	}
	__syncthreads();
	// Now we can start the actual growth
	const unsigned int lower = x * y * blockIdx.x; // (0, 0, blockIdx.x)
	const unsigned int higher = x - 1 + x * (y - 1) + x * y * blockIdx.x;
	for (unsigned int count = 0; count < iterations; count++) {
		for (unsigned int index = lower + threadIdx.x; index < higher; index += blockDim.x) {
			if (grid[index].getType() == CellType::BLANK || grid[index].hasGrown())
				continue;
			int xCoord = (index - lower) % x;
			int yCoord = (index - lower) / x;

			if (grid[index].getType() == CellType::NEURON) {
				if (blockIdx.x == 0) {
					printf("Neuron at (%d, %d)\n", xCoord, yCoord);
				}
				Direction gate = grid[index].getGate();
				bool axonOnX = (gate == Direction::EAST || gate == Direction::WEST);
				// (Try to grow on (i-1, j)
				if (xCoord > 0 && grid[index - 1].getType() == CellType::BLANK) {
					// Attempt to lock (if lock fails then another thread is writing to it)
					if (lock(&(locks[xCoord - 1 + x * yCoord]))) {
						// Lock was successful
						grid[index - 1].growCell(axonOnX ? CellType::AXON : CellType::DENDRITE, Direction::WEST);
						unlock(&(locks[xCoord - 1 + x * yCoord]));
					}
				}

				// Try to grow on (i+1, j)
				if (xCoord < x - 1 && grid[index + 1].getType() == CellType::BLANK) {
					if (lock(&(locks[xCoord + 1 + x * yCoord]))) {
						// Lock was successful
						grid[index + 1].growCell(axonOnX ? CellType::AXON : CellType::DENDRITE, Direction::EAST);
						unlock(&(locks[xCoord + 1 + x * yCoord]));
					}
				}

				// Try to grow on (i, j-1)
				if (yCoord > 0 && grid[index - x].getType() == CellType::BLANK) {
					if (lock(&(locks[xCoord + x * (yCoord - 1)]))) {
						// Lock was successful
						grid[index - x].growCell(axonOnX ? CellType::DENDRITE : CellType::AXON, Direction::NORTH);
						unlock(&(locks[xCoord + x * (yCoord - 1)]));
					}
				}

				// Try to grow on (i, j + 1)
				if (yCoord < y - 1 && grid[index + x].getType() == CellType::BLANK) {
					if (lock(&(locks[xCoord + x * (yCoord + 1)]))) {
						//Lock was successful
						grid[index + x].growCell(axonOnX ? CellType::DENDRITE : CellType::AXON, Direction::SOUTH);
						unlock(&(locks[xCoord + x * (yCoord + 1)]));
					}
				}
			}
			else {
				CellType type = grid[index].getType();
				Direction growthDirection = (type == CellType::AXON) ? grid[index].getGate() : Cell::oppositeDirection(grid[index].getGate());
				Instruction instruction = grid[index].getInstruction();

				// Determine directions to grow into
				unsigned char growthMask = Direction::NONE;
				if ((instruction & (Instruction::GROW_STRAIGHT | Instruction::SPLIT_LEFT | Instruction::SPLIT_RIGHT)) != 0) {
					growthMask |= growthDirection;
				}
				if ((instruction & (Instruction::SPLIT_LEFT | Instruction::TURN_LEFT)) != 0) {
					growthMask |= Cell::rotate(growthDirection, false);
				}
				if ((instruction & (Instruction::SPLIT_RIGHT | Instruction::TURN_RIGHT)) != 0) {
					growthMask |= Cell::rotate(growthDirection, true);
				}

				// Grow into selected direction(s)
				if ((growthMask & Direction::NORTH) != 0) {
					if (yCoord > 0 && grid[index - x].getType() == CellType::BLANK) {
						if (lock(&(locks[xCoord + x * (yCoord - 1)]))) {
							grid[index - x].growCell(type, Direction::NORTH);
							unlock(&(locks[xCoord + x * (yCoord - 1)]));
						}
					}
				}
				if ((growthMask & Direction::EAST) != 0) {
					if (xCoord < x - 1 && grid[index + 1].getType() == CellType::BLANK) {
						if (lock(&(locks[xCoord + 1 + x * yCoord]))) {
							grid[index + 1].growCell(type, Direction::EAST);
							unlock(&(locks[xCoord + 1 + x * yCoord]));
						}
					}
				}
				if ((growthMask & Direction::SOUTH) != 0) {
					if (yCoord < y - 1 && grid[index + x].getType() == CellType::BLANK) {
						if (lock(&(locks[xCoord + x * (yCoord + 1)]))) {
							grid[index + x].growCell(type, Direction::SOUTH);
							unlock(&(locks[xCoord + x * (yCoord + 1)]));
						}
					}
				}
				if ((growthMask & Direction::WEST) != 0) {
					if (xCoord > 0 && grid[index - 1].getType() == CellType::BLANK) {
						if (lock(&(locks[xCoord - 1 + x * yCoord]))) {
							grid[index - 1].growCell(type, Direction::WEST);
							unlock(&(locks[xCoord - 1 + x * yCoord]));
						}
					}
				}
			}
			grid[index].setGrown();
		}
		__syncthreads();
	}
}

__global__ void printKernel(Cell* cells, unsigned int x, unsigned int y, unsigned int z) {
	for (int k = 0; k < z; k++) {
		printf("Slice %d\n", k);
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				printf("%d ", (int)cells[i + x * j + x * y * k].getType());
			}
			printf("\n");
		}
	}
}

void generateRandomNumbers(size_t num, unsigned int* values) {
	random_device rd;
	cout << "Entropy: " << rd.entropy() << endl;

	for (size_t i = 0; i < num; i++) {
		values[i] = rd();
	}
}

void writeChromosome(unsigned int block, unsigned int size, unsigned int * values) {
	fstream fs;
	fs.open("chromosome.h", fstream::out | fstream::trunc);
	fs << "unsigned char chromosome[] = {";
	for (unsigned int i = 0; i < size; i++) {
		if (i % block == 0) {
			fs << endl;
		}
		fs << values[i] % Instruction::I_INVALID << ", ";
	}
	fs << endl << "};";
	fs.close();
}

int main(int argc, char** argv) {
	unsigned int* values = new unsigned int[100 * 100 * 100];
	generateRandomNumbers(100 * 100 * 100, values);
	writeChromosome(100, 100 * 100 * 100, values);
	delete[] values;
	
	/*size_t x, y, z;
	if (argc == 2) {
		x = y = z = atoi(argv[1]);
	}
	else if (argc == 4) {
		x = atoi(argv[1]);
		y = atoi(argv[2]);
		z = atoi(argv[3]);
	}
	else {
		cerr << "Unsupported number of arguments (" << argc - 1 << ")" << endl;
		return 1;
	}*/

	/*Cell* cells = (Cell *)malloc(x * y * z * sizeof(Cell));
	Cell* device_cells = NULL;
	random_device rd;
	// Initialize cells.
	for (unsigned int i = 0; i < x * y * z; i++) {
		// Will this be a neuron?
		bool neuron = (rd() % sig) < (unsigned int)(neuronSeedP * sig);
		// These lines initialize Cell objects continuouslyl in memory (so we can actually memcpy them)
		if ((rd() % sig) < (unsigned int)(neuronSeedP * sig)) {
			new (&(cells[i])) Cell(CellType::NEURON, (Instruction)(char) pow(2, chromosome[i]), Direction::NORTH);
		}
		else {
			new (&(cells[i])) Cell((Instruction)(char) pow(2, chromosome[i]));
		}
	}

	// Allocate cuda cell space
	cudaMalloc((void**)& device_cells, x * y * z * sizeof(Cell));
	// Copy initialized grid to device
	cudaMemcpy(device_cells, cells, x * y * z * sizeof(Cell), cudaMemcpyHostToDevice);

	// Growth Stuff
	auto start_time = chrono::high_resolution_clock::now();
	growKernel <<<z, 64, x * y * sizeof(int) >>> (
		device_cells,
		x,
		y,
		z,
		100
		);
	cudaDeviceSynchronize();
	//printKernel << <1, 1 >> > (device_cells, x, y, z);
	auto end_time = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
	cout << "Time: " << duration << endl;
	cudaMemcpy(cells, device_cells, x * y * z * sizeof(Cell), cudaMemcpyDeviceToHost);
	for (int k = 0; k < z; k++) {
		cout << "Slice " << k << endl;
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				cout << cells[i + x * j + x * y * k].type << " ";
			}
			cout << endl;
		}
	}
	cudaFree(device_cells);
	free(cells);*/

	return 0;
}
