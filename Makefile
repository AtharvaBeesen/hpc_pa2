MPICC = mpicxx
CFLAGS = -Wall -O3 -std=c++2a
MPIRUN = mpirun

many2many: 
	$(MPICC) $(CFLAGS) -o m2m manytomany_main.cpp custom_collectives.cpp

many2many_test1: many2many
	$(MPIRUN) -np 15 ./m2m inputm1.txt outputm1.txt

many2many_test2: many2many
	$(MPIRUN) -np 24 ./m2m inputm2.txt outputm2.txt

allreduce:
	$(MPICC) $(CFLAGS) -o ard allreduce_main.cpp custom_collectives.cpp

allreduce_test1: allreduce
	$(MPIRUN) -np 15 ./ard inputa1.txt outputa1.txt

allreduce_test2: allreduce
	$(MPIRUN) -np 24 ./ard inputa2.txt outputa2.txt

clean:
	rm -f m2m ard

cleanautograder:
	rm -f autograderdata/*.txt



