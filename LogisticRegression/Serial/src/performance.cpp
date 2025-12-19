#include "performace.h"
#if defined(__APPLE__) || defined(__linux__)
#include "sys/resource.h"
#elif defined(_WIN32)
    #include <windows.h>
    #include <psapi.h>
#endif





// fetch memory usage
double getMemoryUsageMB()
{
#if defined(__APPLE__) || defined(__linux__)

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);


    #if defined(__APPLE__)
     // macOS: ru_maxrss is in bytes
        return usage.ru_maxrss / (1024.0 * 1024.0);
    #else
    // Linux: ru_maxrss is in kilobytes
        return usage.ru_maxrss / 1024.0;
    #endif

    #elif defined(_WIN32)

        PROCESS_MEMORY_COUNTERS memCounter;
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            &memcounter,
            sizeof(memCounter)
        );

        return memCounter.WorkingSetSize / (1024.0 * 1024.0);

    #else

    // fallback for unsupported platforms
    return 0.0;
    #endif
    
}