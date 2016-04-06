
bbseq: bbseq.o libbb.o
	mpicxx -O3 bbseq.o libbb.o -o bbseq
	
bbseq.o: bbseq.cc
	mpicxx -O3 -c bbseq.cc


libbb.o: libbb.cc libbb.h
	mpicxx -O3 -c  libbb.cc 


clean:
	/bin/rm -f *.o bbseq



