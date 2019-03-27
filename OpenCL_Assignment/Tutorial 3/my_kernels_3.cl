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