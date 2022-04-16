#include "chronoGPU.hpp"
#include <helper_cuda.h>
#include <iostream>

ChronoGPU::ChronoGPU() 
	: m_started( false ) {
		checkCudaErrors( cudaEventCreate( &m_start ) );
		checkCudaErrors( cudaEventCreate( &m_end ) );
}

ChronoGPU::~ChronoGPU() {
	if ( m_started ) {
		stop();
		std::cerr << "ChronoGPU::~ChronoGPU(): hrono wasn't turned off!" << std::endl; 
	}
	checkCudaErrors( cudaEventDestroy( m_start ) );
	checkCudaErrors( cudaEventDestroy( m_end ) );
}

void ChronoGPU::start() {
	if ( !m_started ) {
		checkCudaErrors( cudaEventRecord( m_start, 0 ) );
		m_started = true;
	}
	else
		std::cerr << "ChronoGPU::start(): chrono is already started!" << std::endl;
}

void ChronoGPU::stop() {
	if ( m_started ) {
		checkCudaErrors( cudaEventRecord( m_end, 0 ) );
		checkCudaErrors( cudaEventSynchronize( m_end ) );
		m_started = false;
	}
	else
		std::cerr << "ChronoGPU::stop(): chrono wasn't started!" << std::endl;
}

float ChronoGPU::elapsedTime() { 
	float time = 0.f;
	checkCudaErrors( cudaEventElapsedTime( &time, m_start, m_end ) );
	return time;
}
