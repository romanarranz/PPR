#include <fstream>
#include <string>
#include <cstdlib>

using namespace std;

int main(int argc, char * argv []){

	int vertices;
	ofstream file;
	int aleatorio;
	string nombre="input";

	vertices=atoi(argv[1]);

	nombre+=string(argv[1]);

	int matriz [vertices][vertices];

	for(int i=0;i<vertices;i++){

		for(int j=0;j<vertices;j++){

			if(i==j){
				matriz[i][j]=-1;
			}
			else{

				aleatorio=(rand() % (vertices*2)) + 1; 
				// (25% de huecos)
				if ( (aleatorio> (vertices*11/20)) ||(aleatorio< (vertices*9/20))){
					 aleatorio=0;
				}				
				matriz[i][j]=aleatorio;
			}

		}
	}

	file.open(nombre.c_str());

	file << vertices << endl;
	
	for(int i=0;i<vertices;i++){

		for(int j=0;j<vertices;j++){
			if(matriz[i][j]>0){
				file << i << " " << j << " " << matriz[i][j] << endl;	
			}
		}
	}
	
	file.close();
	
	return 0;
}
