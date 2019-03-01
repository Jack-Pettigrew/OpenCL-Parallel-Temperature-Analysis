//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
kernel void add(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}

// Reduce: Sum of all Vector Elements
kernel void reduce_sum(global const int* A, global int* B, local int* scratch)
{
	int id = get_global_id(0);			// Global Element Workgroup ID
	int local_id = get_local_id(0);		// Local Element Workgroup ID
	int N = get_local_size(0);			// Local Element Input Size

	// Store into local memory
	scratch[local_id] = A[id];

	// Wait for Global to Local memory complete
	barrier(CLK_LOCAL_MEM_FENCE);

	/*
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

	// Add each Thread result together via Atomic_Add into Output Vector
	if (!local_id) {
		atomic_add(&B[0], scratch[local_id]);
	}
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
