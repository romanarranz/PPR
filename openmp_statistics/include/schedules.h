#ifndef SCHEDULES_H
#define SCHEDULES_H

int f(int i);
double schedule_static_ciclic(int * v, int N, const int k);
double schedule_static_blocks(int * v, int N, const int k);
double schedule_dynamic(int * v, int N, const int k);
double schedule_guided(int * v, int N, const int k);

#endif
