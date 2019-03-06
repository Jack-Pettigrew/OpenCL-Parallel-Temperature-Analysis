#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <iostream>
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
- Optimised Kernels using real temperature values
- Program performance well reported and interpreted
- Well commented code

   1st
- Basic + Median-based statistics on real temp values FLOAT VALUES
- Local memory optimisations considered
- Program performance clearly interpreted in detail
- Optimised, efficient, well-structured, commented in detail
*/

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0;}
	}

	//detect any potential exceptions
	try {

#pragma region Setup

		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		// Load kernel to sources
		AddSources(sources, "my_kernels_1.cl");

		// Create program from Context and Sources
		cl::Program program(context, sources);

		//build and debug the kernel code
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
		
		typedef int myType;
		
		// ==============  Read temperature file into String Vector  ==============
		
		std::vector<string> temperatureInfo;	// Holds file text
		std::vector<myType> temperatureValues;	// Holds all Temperature Floats

		// fstream variables
		ifstream file;
		string fileDir, word;


		fileDir = "temp_lincolnshire_short.txt";
		file.open(fileDir);

		if (!file.is_open())
			cout << "File was not found!" << endl;

		// Read entire file for all Temperature Info
		while (file >> word)
		{
			temperatureInfo.push_back(word);
		}

		// ==============  Extract Floats from String Vector  ==============

		for (int i = 5; i < temperatureInfo.size(); i += 6)
		{

			float temp = strtof(temperatureInfo[i].c_str(), 0);

			temp *= 10;

			temperatureValues.push_back(temp);
		}

		int numOfElements = temperatureValues.size();

		// ==============  Memory Allocation  ==============

		size_t local_size = 32;												// Workgroup size (Too large = CL Size Errors)

		size_t padding_size = temperatureValues.size() % local_size;		// Amount of appenable elements (size_of_vector % workgroup_size)

		/*
			Workgroup Size Handling:

			If the Workgroup size is larger than the ammount of elements taken from the Input...
			fill with empty elements to make up the size difference
		*/
		if (padding_size)
		{
			std::vector<myType> temperature_append(local_size - padding_size, 0);

			temperatureValues.insert(temperatureValues.end(), temperature_append.begin(), temperature_append.end());
		}
		
		size_t input_elements = temperatureValues.size();					// number of elements
		size_t input_size = temperatureValues.size() * sizeof(myType);		// size in bytes
		size_t nr_group = input_elements / local_size;						// total number of workgroups to occur


		// ==============  Host Output Vector ==============

		// Returned values info
		std::vector<myType> B(input_elements);			// Vector B temperature
		std::vector<myType> B_min(input_elements);
		size_t output_size = B.size() * sizeof(myType);	// output vector size in bytes


		// ==============  Device Buffers  ==============

		// Creates Buffers for transfering Input and Output Vectors
		cl::Buffer buffer_temperature_floats(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_B_min(context, CL_MEM_READ_WRITE, output_size);


		// ==============  Device Operations  ==============

		// Copy float vector to device memory
		queue.enqueueWriteBuffer(buffer_temperature_floats, CL_TRUE, 0, input_size, &temperatureValues[0]);
		
		// Initialise Output to device memeory
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_B_min, 0, 0, output_size);

		// Create Kernel call + Set Args
		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_sum");
		kernel_1.setArg(0, buffer_temperature_floats);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size * sizeof(myType)));	// Local Workgroup memory size
		
		cl::Kernel kernel_2 = cl::Kernel(program, "reduce_min");
		kernel_2.setArg(0, buffer_temperature_floats);
		kernel_2.setArg(1, buffer_B_min);
		kernel_2.setArg(2, cl::Local(local_size * sizeof(myType)));

		// Queue Kernel calls
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));	// Apply custom workgroup size

		// Get results openCL device
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
		queue.enqueueReadBuffer(buffer_B_min, CL_TRUE, 0, output_size, &B_min[0]);
	

		float sum = B[0];
		sum /= 10;
		float avg = sum / numOfElements;
		float min_value = B_min[0] / 10;

		// Output Result
		std::cout << "Sum = " << sum << endl;
		std::cout << "Average = " << avg << endl;
		std::cout << "Min= " << min_value << endl;

	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
