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

/*	Sort Input into Numerical Order

	As referenced by Eric Bainville: http://www.bealto.com/gpu-sorting_parallel-selection.html
*/
kernel void sort(global const int* A, global int* B, local int* scratch)
{
	int id = get_global_id(0);      // Global ID
	int local_id = get_local_id(0);	// Local ID
	int N = get_global_size(0);     // Input size
	int wg = get_local_size(0);     // Workgroup size
	int iKey = A[id];				// Input key for current thread

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
			int jKey = scratch[index];
			bool smaller = (jKey < iKey) || (jKey == iKey && (i + index) < id); // in[j] < in[i] ?
			pos += (smaller) ? 1 : 0;
		}
	}

	B[pos] = iKey;

}

kernel void sort_mine(global const int* A, global int* B, local int* scratch)
{
	int id = get_global_id(0);      // Global ID
	int N = get_global_size(0);     // Input size

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride * 2)) == 0)
		{
			if (A[id - stride] > A[id])
			{
				scratch[id] = A[id - stride];
				scratch[id - stride] = A[id];
			}
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N / 2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride * 2)) == 0) 
		{
			if (A[id - stride] > A[id])
			{
				scratch[id] = A[id - stride];
				scratch[id - stride] = A[id];
			}
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	B[id] = scratch[id];

}

// Returns Vector containing Standard Deviation of the input set (A)
kernel void std_dev(global const int* A, global int* B, global const int* sum, local int* scratch)
{
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int N = get_local_size(0);

	// Mean
	int avg = sum[0] / get_global_size(0);

	// Copy Input per Workgroup Item
	scratch[local_id] = ((A[id] - avg) * (A[id] - avg)) / 10;

	barrier(CLK_LOCAL_MEM_FENCE);

	// Reduction Addition
	for (int i = 1; i < N; i *= 2)
	{
		if (!(local_id % (i * 2)) && ((local_id + i) < N))
		{
			scratch[local_id] += scratch[local_id + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!local_id)
		atomic_add(&B[0], scratch[local_id]);
}

//a simple smoothing kernel averaging values in a local window (radius 1)
kernel void avg_filter(global const int* A, global int* B) {
	int id = get_global_id(0);
	B[id] = (A[id - 1] + A[id] + A[id + 1])/3;
}

//a simple 2D kernel
kernel void add2D(global const int* A, global const int* B, global int* C) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int id = x + y*width;

	printf("id = %d x = %d y = %d w = %d h = %d\n", id, x, y, width, height);

	C[id]= A[id]+ B[id];
}

// Min value finding Kernel
kernel void reduce_minValue(global float* A, global float* B)
{
	int id = get_global_id(0);	// Current Workgroup ID
	int N = get_local_size(0);  // Number of local item in current Workgroup

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);
	
	for (int stride = 1; stride < N; stride *= 2)
	{
		if ((id % (stride * 2)) == 0 && (id + 1) < N)
		{
			B[id] = ((B[id] > B[id + stride]) ? B[id] : B[id + stride]);
		}
			
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}
