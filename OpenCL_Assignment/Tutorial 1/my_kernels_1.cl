// Reduce Sum of all Vector Elements from vector A to B using a local memory Vector scratch
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

	// Part 3: Calculate total for each Workgroup
	if (!local_id) {
		B[g_id] = scratch[local_id];
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

