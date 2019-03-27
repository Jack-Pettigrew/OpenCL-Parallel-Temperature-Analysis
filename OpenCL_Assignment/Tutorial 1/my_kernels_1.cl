// a simple OpenCL kernel which adds two vectors A and B together into a third vector C
kernel void add(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}

// Reduce Sum of all Vector Elements from vector A to B using a local memory Vector scratch
kernel void reduce_sum(global const int* A, global int* B, local int* scratch)
{
	int id = get_global_id(0);			// Global Element Workgroup ID
	int local_id = get_local_id(0);		// Local Element Workgroup ID
	int N = get_local_size(0);			// Size of Local Workgroup

	// Part 1: Store into local memory
	scratch[local_id] = A[id];

	// Wait for Global to Local memory complete
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Part 2:
		Automated Stride Loop:

		Stride = i (doubled each step) 1 -> 2 -> 4 -> 8 -> 16 etc...

		Add results into Cached Vector and then return the Cached vector to the output vector B
	*/
	for (int i = 1; i < N; i *= 2)
	{
		if (!(local_id % (i * 2)) && ((local_id + i) < N))
		{
			scratch[local_id] += scratch[local_id + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Part 3: Add each Thread result together via Atomic_Add into Output Vector
	if (!local_id) {
		atomic_add(&B[0], scratch[local_id]);
	}
}

kernel void reduce_sum_float(global const float* A, global float* B, local float* scratch)
{
	int id = get_global_id(0);			// Global Element Workgroup ID
	int local_id = get_local_id(0);		// Local Element Workgroup ID
	int N = get_local_size(0);			// Size of Local Workgroup
	int g_id = get_group_id(0);			// Workgroup ID

	// Part 1: Store into local memory
	scratch[local_id] = A[id];

	// Wait for Global to Local memory complete
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int stride = N / 2; stride > 0; stride /= 2) 
	{
		if (local_id < stride)
			scratch[local_id] += scratch[local_id + stride];

		barrier(CLK_LOCAL_MEM_FENCE);

	}

	// Part 3: Add each Thread result together via Atomic_Add into Output Vector
	if (!local_id) {

		B[g_id] = scratch[local_id];

	}
}

// Reduce Min value given in vector A outputted in vector B via local memory vector Scratch
kernel void reduce_min(global const int* A, global int* B, local int* scratch)
{
	int id = get_global_id(0);	
	int local_id = get_local_id(0);
	int N = get_local_size(0);	

	// Part 1: Store into local memory
	scratch[local_id] = A[id];

	// Wait for Global to Local memory complete
	barrier(CLK_LOCAL_MEM_FENCE);

	// Part 2: Automated Stride Loop:
	for (int stride = 1; stride < N; stride *= 2)
	{
		if (!(local_id % (stride * 2)) && ((local_id + stride) < N))
		{
			if (scratch[local_id + stride] < scratch[local_id])
			{
				scratch[local_id] = scratch[local_id + stride];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// Part 3: Atomic_Min
	if (!local_id) {
		atomic_min(&B[0], scratch[local_id]);
	}
}

kernel void reduce_min_float(global const float* A, global float* B, local float* scratch)
{
	int id = get_global_id(0);			// Global Element Workgroup ID
	int local_id = get_local_id(0);		// Local Element Workgroup ID
	int N = get_local_size(0);			// Size of Local Workgroup
	int g_id = get_group_id(0);

	// Part 1: Store into local memory
	scratch[local_id] = A[id];

	// Wait for Global to Local memory complete
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int stride = N / 2; stride > 0; stride /= 2) {

		if (local_id < stride)
			if (scratch[local_id + stride] < scratch[local_id])
				scratch[local_id] = scratch[local_id + stride];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Part 3: Add each Thread result together via Atomic_Add into Output Vector
	if (!local_id) {

		B[g_id] = scratch[local_id];

	}
}

// Reduce Max value given in vector A outputted in vector B via local memory vector Scratch
kernel void reduce_max(global const int* A, global int* B, local int* scratch)
{
	int id = get_global_id(0);			// Global Element Workgroup ID
	int local_id = get_local_id(0);		// Local Element Workgroup ID
	int N = get_local_size(0);			// Local Element Input Size

	// Part 1: Store into local memory
	scratch[local_id] = A[id];

	// Wait for Global to Local memory complete
	barrier(CLK_LOCAL_MEM_FENCE);

	// Part 2: Automated Stride Loop:
	for (int stride = 1; stride < N; stride *= 2)
	{
		if (!(local_id % (stride * 2)) && ((local_id + stride) < N))
		{
			if (scratch[local_id + stride] > scratch[local_id])
			{
				scratch[local_id] = scratch[local_id + stride];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Part 3: Atomic_Min
	if (!local_id) {
		atomic_max(&B[0], scratch[local_id]);
	}
}

kernel void reduce_max_float(global const float* A, global float* B, local float* scratch)
{
	int id = get_global_id(0);			// Global Element Workgroup ID
	int local_id = get_local_id(0);		// Local Element Workgroup ID
	int N = get_local_size(0);			// Size of Local Workgroup
	int g_id = get_group_id(0);

	// Part 1: Store into local memory
	scratch[local_id] = A[id];

	// Wait for Global to Local memory complete
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int stride = N / 2; stride > 0; stride /= 2) {

		if (local_id < stride)
			if (scratch[local_id + stride] > scratch[local_id])
				scratch[local_id] = scratch[local_id + stride];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Part 3: Add each Thread result together via Atomic_Add into Output Vector
	if (!local_id) {

		B[g_id] = scratch[local_id];

	}
}

/*	Sort Input into Numerical Order

	As referenced by Eric Bainville: http://www.bealto.com/gpu-sorting_parallel-selection.html
*/
kernel void sort(global const float* A, global float* B, local float* scratch)
{
	int id = get_global_id(0);      // Global ID
	int local_id = get_local_id(0);	// Local ID
	int N = get_global_size(0);     // Input size
	int wg = get_local_size(0);     // Workgroup size
	float iKey = A[id];				// Input key for current thread

	// Output index position
	int pos = 0;
	for (int i = 0; i < N ; i += wg)
	{
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int index = local_id; index < wg; index += wg)
		{
			scratch[index] = A[i + index];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop on all values in Scratch
		for (int index = 0; index < wg; index++)
		{
			float jKey = scratch[index];
			bool smaller = (jKey < iKey) || (jKey == iKey && (i + index) < id); // in[j] < in[i] ?
			pos += (smaller) ? 1 : 0;
		}
	}

	B[pos] = iKey;

}

// LECTURE SORT ATTEMPT
//// Compares value A with B and exchanges if unordered
//void cmpxchg(global int* A, global int* B) 
//{
//	
//	if (*A > *B) {
//		// Swap
//		int t = *A; 
//		*A = *B; 
//		*B = t;
//	}
//
//}
//
//// Checks each element is ordered in Odd/Even rotation
//kernel void sort_oddeven(global int* A) 
//{
//	int id = get_global_id(0); 
//	int N = get_global_size(0);
//
//	for (int i = 0; i < N; i += 2)
//	{
//		if (id % 2 == 1 && id + 1 < N)	// Odd Step
//			cmpxchg(&A[id], &A[id + 1]);
//
//		if (id % 2 == 0 && id + 1 < N)	// Even Step
//			cmpxchg(&A[id], &A[id + 1]);
//	}
//}

/*  BITONIC SORT ATTEMPT

// Compares value A with B and exchanges in Ascend/Descend rotation
void cmpxchg(global int* A, global int* B, bool dir) 
{
	if ((!dir && *A > *B) || (dir && *A < *B)) {
		int t = *A;
		*A = *B;
		*B = t;
	}
}

// Merge Bitonic sequences
void bitonic_merge(int id, global int* A, int N, bool dir)
{
	// Split into bitonic sequences each iteration
	for (int i = N/2; i > 0; i /= 2)
	{

		if ((id % (id * 2)) < i)
			cmpxchg(&A[id], &A[id + 1], dir);

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

// Conducts Sorting algorithm via Bitonic Sort
kernel void bitonic_sort(global int* A)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int i = 1; i < N / 2; i *= 2)
	{
		if (id % (i * 4) < i * 2)
			bitonic_merge(id, A, i * 2, false);	// Bitonic Ascending

		else if ((id + i * 2) % (i * 4) < i * 2)
			bitonic_merge(id, A, i * 2, true);	// Bitonic Descending

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	if (id == 0)
		bitonic_merge(id, A, N, false);			// Final Merge

}

*/

// Returns Vector containing the Standard Deviation of the Sum input
kernel void std_dev(global const int* A, global int* B, global const int* sum, local int* scratch)
{
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int N = get_local_size(0);

	// Calculate Mean (A = Sum Output)
	int avg = sum[0] / get_global_size(0);

	// Copy the square of each distance to the Mean into Local_Mem
	scratch[local_id] = ((A[id] - avg) * (A[id] - avg)) / 10;


	// Sync
	barrier(CLK_LOCAL_MEM_FENCE);


	// Reduce Addition all differences
	for (int i = 1; i < N; i *= 2)
	{
		if (!(local_id % (i * 2)) && ((local_id + i) < N))
		{
			scratch[local_id] += scratch[local_id + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Atomically Add all Workgroup Local Additions
	if (!local_id)
		atomic_add(&B[0], scratch[local_id]);

}

kernel void std_dev_float(global const float* A, global float* B, global const float* sum, local float* scratch)
{
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int N = get_local_size(0);
	int g_id = get_group_id(0);

	// Calculate Mean (A = Sum Output)
	float avg = sum[0] / get_global_size(0);

	// Copy the square of each distance to the Mean into Local_Mem
	scratch[local_id] = ((A[id] - avg) * (A[id] - avg));


	// Sync
	barrier(CLK_LOCAL_MEM_FENCE);


	// Reduce Addition all differences
	for (int i = 1; i < N; i *= 2)
	{
		if (!(local_id % (i * 2)) && ((local_id + i) < N))
		{
			scratch[local_id] += scratch[local_id + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Atomically Add all Workgroup Local Additions
	if (!local_id)
		B[g_id] = scratch[local_id];

}

