#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <stdlib.h>

#include "Utils.h"

/* Pass
- Basic Summary of weather data ( min/max/avg/standard deviation ) INTEGER VALUES
- Memory transfer times are provided
- Readable Code

   2:2
- Attempt to optimise code using INTERGER VALUES
- Program Performance Provided
- Clear Coding style with comments

   2:1
- Optimised Kernels using real temperature values FLOAT VALUES
- Program performance well reported and interpreted
- Well commented code

   1st
- Basic + Median-based statistics on real temp values FLOAT VALUES
- Local memory optimisations considered
- Program performance clearly interpreted in detail
- Optimised, efficient, well-structured, commented in detail
*/

// Launch Arguments (e.g. "Tutorial1 - p")
void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv)
{

	// Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	// Try loop entire Parallel Code for Errors
	try {


		// OpenCL Init procedure
#pragma region Setup

		// Part 2 - host operations

		// 2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		// Display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		// Create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// 2.2 Load & build the device code
		cl::Program::Sources sources;

		// Load kernel to sources
		AddSources(sources, "my_kernels_1.cl");

		// Create program from Context + Sources
		cl::Program program(context, sources);

		// Build + Debug the Kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

#pragma endregion

		// Value Types
		typedef int myType;


		// ==============  Read temperature file into String Vector  ==============

		std::vector<string> temperatureInfo;	/// Holds file text
		std::vector<myType> temperatureValues;	/// Holds all Temperature Floats

		// File reading variables
		ifstream file;
		string fileDir, word;

		// Directory of Temperature Files
		/// Relative Pathing
		//fileDir = "..\\..\\temp_lincolnshire_short.txt";
		//fileDir = "..\\..\\temp_lincolnshire.txt";

		/// Aboslute Pathing
		//fileDir = "C:\\Users\\Student\\Desktop\\OpenCL-Assignment\\OpenCL_Assignment\\temp_lincolnshire_short.txt";
		fileDir = "C:\\Users\\Student\\Desktop\\OpenCL-Assignment\\OpenCL_Assignment\\temp_lincolnshire.txt";

		file.open(fileDir);

		if (!file.is_open())
			cout << "\nTemperature file was not found!" << endl;

		// Read fileDir for raw input
		while (file >> word)
		{
			temperatureInfo.push_back(word);
		}

		// ==============  Extract Floats from String Vector  ==============


		// Extract temp values from raw Vector
		for (int i = 5; i < temperatureInfo.size(); i += 6)
		{
			float temp = strtof(temperatureInfo[i].c_str(), 0);
			temp * 10;
			temperatureValues.push_back(temp);
		}

		// Used to calculate Average
		int numOfElements = temperatureValues.size();



		// ==============  Memory Allocation  ==============

		size_t local_size = 64;												// OpenCL device Workgroup size (Non-multiple = CL_ERRORS)

		size_t padding_size = temperatureValues.size() % local_size;		// Amount of appenable elements ('0')

		/* Workgroup Size Handling (Padding):

			If the Workgroup size is larger than the ammount of input elements...
			...fill with empty elements to make up the size difference
		*/
		if (padding_size)
		{
			std::vector<myType> temperature_append(local_size - padding_size, 0);
			temperatureValues.insert(temperatureValues.end(), temperature_append.begin(), temperature_append.end());
		}

		// OpenCL data values
		size_t input_elements = temperatureValues.size();					// number of elements
		size_t input_size = temperatureValues.size() * sizeof(myType);		// size in bytes
		size_t nr_group = input_elements / local_size;						// total number of workgroups to occur



		// ==============  Host Output Vector ==============

		// Returned values info
		std::vector<myType> B_sum(input_elements);
		std::vector<myType> B_min(input_elements);
		std::vector<myType> B_max(input_elements);
		std::vector<myType> B_sort(input_elements);
		std::vector<myType> B_std(input_elements);

		// Resulting Vector Size
		size_t output_size = B_sum.size() * sizeof(myType);



		// ==============  Device Buffers  ==============

		// Creates Buffers Input and Output Vectors
		// Buffer A
		cl::Buffer buffer_temperatures(context, CL_MEM_READ_WRITE, input_size);

		// Buffer B(s)
		cl::Buffer buffer_B_sum(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_B_min(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_B_max(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_B_sort(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_B_std(context, CL_MEM_READ_WRITE, output_size);



		// ==============  Device Operations  ==============

		// Create device input temperature vector Buffer
		queue.enqueueWriteBuffer(buffer_temperatures, CL_TRUE, 0, input_size, &temperatureValues[0]);

		// Create device output vector Buffer for each test case
		queue.enqueueFillBuffer(buffer_B_sum, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_B_min, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_B_max, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_B_sort, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_B_std, 0, 0, output_size);



		// ============== Sum FLOATS ==============
		/// Returns the sum of all values in the first element of the vector

		// Create Profiling Event (kernel information)
		cl::Event profiling_event;

		// Create Kernel call and set arguements
		cl::Kernel kernel_sum = cl::Kernel(program, "reduce_sum");
		kernel_sum.setArg(0, buffer_temperatures);							/// Input Vector Read Buffer
		kernel_sum.setArg(1, buffer_B_sum);									/// Output Vector Write Buffer
		kernel_sum.setArg(2, cl::Local(local_size * sizeof(myType)));		/// Local Memory size value

		// Queue 
		queue.enqueueNDRangeKernel(kernel_sum, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &profiling_event);
		queue.enqueueReadBuffer(buffer_B_sum, CL_TRUE, 0, output_size, &B_sum[0]);



		// ============== Min Value FLOATS ==============
		/// Returns Max value in input vector and stores it in the first element


		cl::Event profiling_min;

		cl::Kernel kernel_min = cl::Kernel(program, "reduce_min");
		kernel_min.setArg(0, buffer_temperatures);
		kernel_min.setArg(1, buffer_B_min);
		kernel_min.setArg(2, cl::Local(local_size * sizeof(myType)));

		queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &profiling_min);
		queue.enqueueReadBuffer(buffer_B_min, CL_TRUE, 0, output_size, &B_min[0]);


		// ============== Max Value FLOATS ==============
		/// Returns Max value in input vector and stores it in the first element

		cl::Event profiling_max;

		cl::Kernel kernel_max = cl::Kernel(program, "reduce_max");
		kernel_max.setArg(0, buffer_temperatures);
		kernel_max.setArg(1, buffer_B_max);
		kernel_max.setArg(2, cl::Local(local_size * sizeof(myType)));

		queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &profiling_max);
		queue.enqueueReadBuffer(buffer_B_max, CL_TRUE, 0, output_size, &B_max[0]);



		// ============== STD Deviation ==============
		/// Calculates the first steps of Standard Deviation

		cl::Event profiling_std;

		cl::Kernel kernel_std = cl::Kernel(program, "std_dev");
		kernel_std.setArg(0, buffer_temperatures);
		kernel_std.setArg(1, buffer_B_std);
		kernel_std.setArg(2, buffer_B_sum);
		kernel_std.setArg(3, cl::Local(local_size * sizeof(myType)));

		queue.enqueueNDRangeKernel(kernel_std, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &profiling_std);
		queue.enqueueReadBuffer(buffer_B_std, CL_TRUE, 0, output_size, &B_std[0]);



		// ============== Format Results ==============
		float sum = B_sum[0] / 10.0f;
		float avg = sum / numOfElements;
		float min_value = (float)B_min[0] / 10.0f;
		float max_value = (float)B_max[0] / 10.0f;
		float variance = (B_std[0] / B_std.size()) / 10.0f;
		float std_dev = sqrt(variance);



		// ==============  Output Results + Profiling  ==============

		std::cout << "\nProgram Execution Completed!\n" << endl;

		std::cout << GetFullProfilingInfo(profiling_event, ProfilingResolution::PROF_US) << endl;
		std::cout << "Workgroup Size: " << local_size << endl << endl;

		std::cout << "********************* INT Results *********************" << endl;
		std::cout << "Sum		= " << sum << endl;
		std::cout << "Average		= " << avg << endl;
		std::cout << "Min		= " << min_value << endl;
		std::cout << "Max		= " << max_value << endl;
		std::cout << "Std Deviation   = " << std_dev << endl << endl;

		std::cout << "********************* Profiling *********************" << endl;
		std::cout << "AVG Time:	" << profiling_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profiling_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns]" << endl;
		std::cout << "Min Time:	" << profiling_min.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profiling_min.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns]" << endl;
		std::cout << "Max Time:	" << profiling_max.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profiling_max.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns]" << endl;
		std::cout << "Std Time:	" << profiling_std.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profiling_std.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " [ns]" << endl << endl;
		///std::cout << "Median finish:	"	<< profiling_sort.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profiling_sort.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " ns (includes Sort)" << endl << endl;

		std::cout << "Total Program Execution Time: " << profiling_max.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profiling_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << " ns \n" << endl;

	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	// Pauses Visual Studio Debug
	system("pause");

	return 0;
}
