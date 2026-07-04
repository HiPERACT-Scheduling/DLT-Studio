//---------------------------------------------------------------------------
// util/time.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//---------------------------------------------------------------------------

#ifndef DLS_UTIL_TIME_H
#define DLS_UTIL_TIME_H

#include <ctime>
#include <chrono>
#include <string>


class Time
{
public:

	time_t  t;
	clock_t clockvalue;
	struct tm *now;
    char       buf[80];
    std::chrono::high_resolution_clock::time_point t_start, t_end;



	Time();
	~Time();
	std::string CurrentDateTime();
	std::string CurrentDateTimeShifted(int);
	int GetYear();
	int GetYDay();
	int GetMonth();
	int GetMDay();
	int GetWDay();
	int GetHour();
	int GetMin();
	int GetSec();
	clock_t GetClock();
	std::chrono::high_resolution_clock::time_point StartDuration();
	double CalcDuration(std::chrono::high_resolution_clock::time_point);
	bool SetStartTime();
	bool SetStopTime();
	double CalcWallTime();


/*
	time_t t = time(0);   // get time now
    struct tm * now = localtime( & t );
    cout << (now->tm_year + 1900) << '-'
         << (now->tm_mon + 1) << '-'
         <<  now->tm_mday << " "
		 <<  now->tm_wday << " "
		 <<  now->tm_hour << " "
		 <<  now->tm_min << " "
		 <<  now->tm_sec << " "
         <<  endl;
*/

};


#endif // DLS_UTIL_TIME_H
