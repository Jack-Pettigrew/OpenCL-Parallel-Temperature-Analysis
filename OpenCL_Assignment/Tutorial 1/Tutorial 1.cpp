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
		
		
		// ==============  Read temperature file into String Vector  ==============
		
		std::vector<string> temperatureInfo;	// Holds file text
		std::vector<float> temperatureFloats;	// Holds all Temperature Floats

		// fstream variables
		fstream file;
		string fileDir, word;


		fileDir = "temp_lincolnshire_short.txt";
		file.open(fileDir.c_str());


		// Read entire file for all Temperature Info
		while (file >> word)
		{
			temperatureInfo.push_back(word);
		}


		// ==============  Extract Floats from String Vector  ==============

		for (int i = 5; i < temperatureInfo.size(); i += 6)
		{

			float temp = std::stof(temperatureInfo[i].c_str());

			temperatureFloats.push_back(temp);
		}

		//cout << temperatureFloats << endl;


		// ==============  Memory Allocation  ==============

		// Vector size info
		size_t vector_elements = temperatureFloats.size();					// number of elements
		size_t vector_size = temperatureFloats.size() * sizeof(int);		// size in bytes

		// Returned values info
		std::vector<float> min_value(vector_elements);		// minimum temperature


		// ==============  Device Buffers  ==============

		cl::Buffer buffer_temperature_floats(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_min_value(context, CL_MEM_READ_WRITE, vector_size);


		// ==============  Device Operations  ==============

		// Copy float vector to device memory
		queue.enqueueWriteBuffer(buffer_temperature_floats, CL_TRUE, 0, vector_size, &temperatureFloats[0]);

		// Create Kernel call
		cl::Kernel kernel_min = cl::Kernel(program, "reduce_minValue");
		kernel_min.setArg(0, temperatureFloats);
		kernel_min.setArg(1, min_value);

		// Queue Kernel call
		queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange);

		// Get results from device
		queue.enqueueReadBuffer(buffer_min_value, CL_TRUE, 0, vector_size, &min_value);

		// Output Result
		std::cout << "Min Temp: " << min_value << " degrees" << endl;

	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
