#include "environment.h"

Environment::Environment():
	_qtyHives(10), _qtyNestBoxes(20), _step(0), _repetition(0), _generation(0)
{
	// Avoid spawning on corners
	float border = 0.9;

	// Create the nestbox with the qty of boxes
	_nestBoxes = new NestBox[_qtyNestBoxes];

	for(int i=0; i<_qtyNestBoxes; i++)
	{
		// Initialize the nestbox in a random position
		float x = ((rand()%2000)/1000.f-1.0)*border;
		float y = ((rand()%2000)/1000.f-1.0)*border;
		float goodness = (rand()%1000)/1000.f;

		_nestBoxes[i] = NestBox(x, y, goodness);
	}

	for(int i=0; i<_qtyHives; i++)
	{
		// Initialize the hive in a random position
		float x = ((rand()%2000)/1000.f-1.0)*border;
		float y = ((rand()%2000)/1000.f-1.0)*border;

		double* gene = new double[4];
		gene[0] = rand()%100000000/100000000.f;//0.00005;//
		gene[1] = rand()%100000000/100000000.f;//0.3;//
		gene[2] = rand()%100000000/100000000.f;//0.0001;//
		gene[3] = rand()%100000000/100000000.f;//0;//

		float r = rand()%100/100.f;
		float g = rand()%100/100.f;
		float b = rand()%100/100.f;
		
		Hive* hive = new Hive(x, y, gene, r, g, b);
		hive->setNestBoxes(_nestBoxes, _qtyNestBoxes);
		
		_hives.push_back(hive);

		//if(i==0)
		//{
		//	gene[0] = 0.0005f;//rand()%100000000/100000000.f;//0.00005;//
		//	gene[1] = 0.003;//rand()%100000000/100000000.f;//0.3;//
		//	gene[2] = 0.001;//rand()%100000000/100000000.f;//0.0001;//
		//	gene[3] = 0;//rand()%100000000/100000000.f;//0;//

		//	_hives[0]->reset(x, y, gene);
		//}
	}
}

Environment::~Environment()
{
	for(auto hive : _hives)
	{
		delete hive;
	}
	_hives.clear();

	//delete _nestBoxes;
}

void Environment::draw()
{
	for(int i=0; i<_qtyNestBoxes; i++)
		_nestBoxes[i].draw();

	// Find best hive
	//if(_generationFitness.size())
	//int bestIndex = 0;
	//float bestFitness = _generationFitness.back()[0];
	//for(int i=0;i<_qtyHives;i++)
	//{
	//	float fitness = _generationFitness.back()[i];
	//	if(fitness > bestFitness)
	//	{
	//		bestIndex = i;
	//		bestFitness = fitness;
	//	}
	//}
	//_hives[bestIndex]->draw();
	for(auto hive : _hives)
		hive->draw();

	// Draw progress bar
	glColor3f(1,0,0);
	glBegin(GL_POLYGON);
	{
		float progress = (float(_step)/STEPS_PER_REPETITION)*(1.f/(REPETITIONS_PER_GENERATION)) 
			+ (float(_repetition)/REPETITIONS_PER_GENERATION);
		glVertex2d(-1,1);
		glVertex2d((progress-0.5f)*2,1);
		glVertex2d((progress-0.5f)*2,1-(3.0f/MAIN_WINDOW_HEIGHT));
		glVertex2d(-1,1-(3.0f/MAIN_WINDOW_HEIGHT));
	}
	glEnd();
}

void Environment::plotConsensus()
{
	int i=-1;
	float sizeEach = 2.0f/_qtyHives;
	float sizeColorBar = sizeEach*0.10;
	float maxBarSize = sizeEach*0.90;
	for(auto hive : _hives)
	{
		i++;
		float offsetY = -1+i*sizeEach;
		//float ratio = float(PLOT_WINDOW_WIDTH)/PLOT_WINDOW_HEIGHT;
		float offset = 2.0f/(_qtyNestBoxes+1);
		float size = (2.0f/(_qtyNestBoxes))*0.4f;

		int* consensus = hive->getConsensus();
		int qtyScoutBees = hive->getQtyScoutBees();

		std::vector<std::pair<float, int>> orderedNestBoxes;

		for(int i=0; i<_qtyNestBoxes; i++)
		{
			float goodness = _nestBoxes[i].getRealGoodness();
			orderedNestBoxes.push_back(std::make_pair(goodness,i));
		}
		std::sort(orderedNestBoxes.begin(), orderedNestBoxes.end());

		// Plot background color
		glColor3f(hive->getColor(1), hive->getColor(2), hive->getColor(3));
		glBegin(GL_POLYGON);
		{
			glVertex2d(-1, offsetY+sizeColorBar/3);
			glVertex2d(-1, offsetY);
			glVertex2d(1, offsetY);
			glVertex2d(1, offsetY+sizeColorBar/3);
		}
		glEnd();

		// Plot nest boxes
		for(int i=0; i<_qtyNestBoxes; i++)
		{
			float goodness = orderedNestBoxes[i].first;
			glColor3f(goodness, 0, goodness);
			glBegin(GL_POLYGON);
			{
				glVertex2d(-1+offset*(i+1)-size, offsetY+sizeColorBar);
				glVertex2d(-1+offset*(i+1)-size, offsetY);
				glVertex2d(-1+offset*(i+1)+size, offsetY);
				glVertex2d(-1+offset*(i+1)+size, offsetY+sizeColorBar);
			}
			glEnd();

			glColor3f(0, 0, 0);
			glBegin(GL_POLYGON);
			{
				glVertex2d(-1+offset*(i+1)-size*0.5, offsetY+sizeColorBar+maxBarSize*float(consensus[orderedNestBoxes[i].second])/qtyScoutBees);
				glVertex2d(-1+offset*(i+1)-size*0.5, offsetY+sizeColorBar);
				glVertex2d(-1+offset*(i+1)+size*0.5, offsetY+sizeColorBar);
				glVertex2d(-1+offset*(i+1)+size*0.5, offsetY+sizeColorBar+maxBarSize*float(consensus[orderedNestBoxes[i].second])/qtyScoutBees);
			}
			glEnd();
		}
	}
}

void Environment::plotGeneration()
{
	for(int i=1; i<_generation; i++)
	{
		float xPos = 2.f*float(i)/(_generation-1) -1.f;
		float lastXPos = 2.f*float(i-1)/(_generation-1) -1.f;
		for(int j=0; j<_qtyHives; j++)
		{
			glColor3f(0,0,0);
			glColor3f(_hives[j]->getColor(1), _hives[j]->getColor(2), _hives[j]->getColor(3));
			glBegin(GL_LINES);
			{
				glVertex2f(lastXPos, _generationFitness[i-1][j]/100.f*2-1);
				glVertex2f(xPos, _generationFitness[i][j]/100.f*2-1);
			}
			glEnd();
		}
	}
}

void Environment::run(int steps, fstream datafile)
{


	for(auto hive : _hives)
	{
		hive->run(steps);
	}

	_step+=steps;
	//std::cout << _step << "/" << STEPS_PER_GENERATION << std::endl;

	//---------------- Repetition finished -------------------//
	if(_step>=STEPS_PER_REPETITION)
	{

		//data << "Repetition " << _repetition << " finished!" << std::endl;
		// Reset repetition
		_step = 0;
		_repetition++;
		
		// Reset nest boxes
		float border = 0.9;
		for(int i=0; i<_qtyNestBoxes; i++)
		{
			float x = ((rand()%2000)/1000.f-1.0)*border;
			float y = ((rand()%2000)/1000.f-1.0)*border;
			float goodness = (rand()%1000)/1000.f;

			_nestBoxes[i] = NestBox(x, y, goodness);
		}

		// Add fitness to vector
		_repetitionFitness.push_back({});
		for(int i=0; i<_qtyHives; i++)
		{
			_repetitionFitness.back().push_back(_hives[i]->getFitness()*100);
		}
		
		// Reset hives position
		for(int i=0;i<_qtyHives;i++)
		{
			double* gene = _hives[i]->getGene();

			float border = 0.9;
			float x = ((rand()%2000)/1000.f-1.0)*border;
			float y = ((rand()%2000)/1000.f-1.0)*border;

			_hives[i]->reset(x, y, gene);
			_hives[i]->setNestBoxes(_nestBoxes, _qtyNestBoxes);
		}

		//---------------- Generation finished -------------------//
		if(_repetition>=REPETITIONS_PER_GENERATION)
		{
			datafile << "Generation " << _generation << std::endl;
			
			// Reset generation
			_generation++;
			_repetition=0;

			// Calculate fitness
			_generationFitness.push_back({});
			for(int i=0; i<_qtyHives; i++)
			{
				float mean = 0;
				for(int j=0; j<REPETITIONS_PER_GENERATION; j++)
					mean += _repetitionFitness[j][i];
				mean/=REPETITIONS_PER_GENERATION;
				
				datafile << "(" << i << ") fitness = " << mean << std::endl;				

				// Add fitness to vector
				_generationFitness.back().push_back(mean);
			}
			_repetitionFitness.clear();

			// Find best hive
			int bestIndex = 0;
			float bestFitness = _generationFitness.back()[0];
			std::vector<std::pair<float, int>> hivesFitness;
			for(int i=0;i<_qtyHives;i++)
			{
				float fitness = _generationFitness.back()[i];
				hivesFitness.push_back(std::make_pair(fitness,i));
				if(fitness > bestFitness)
				{
					bestIndex = i;
					bestFitness = fitness;
				}
			}
			double* bestGene = _hives[bestIndex]->getGene();
			
			// Cross hives
			for(int i=0;i<_qtyHives;i++)
			{
				double* gene = _hives[i]->getGene();

				if(i!=bestIndex)
					for(int j=0;j<4;j++)
					{
						float mutationForce = 0;
						int random = rand()%5;
						switch(random)
						{
							case 0:
								mutationForce = 100;
								break;
							case 1:
								mutationForce = 10;
								break;
							case 2:
								mutationForce = 1;
								break;
							case 3:
								mutationForce = 0.1;
								break;
							case 4:
								mutationForce = 0.01;
								break;
						}
						mutationForce = 1;

						do
							gene[j] = bestGene[j]*0.5f + gene[j]*0.5f + mutationForce*bestGene[j]*(rand()%1000/1000.0 - 500/1000.0);
						while(gene[j]<0 || gene[j]>1);
					}

				float border = 0.9;
				float x = ((rand()%2000)/1000.f-1.0)*border;
				float y = ((rand()%2000)/1000.f-1.0)*border;

				_hives[i]->reset(x, y, gene);
				_hives[i]->setNestBoxes(_nestBoxes, _qtyNestBoxes);
			}
			
			// Predation
			if(_generation%15==0)
			{
				std::sort(hivesFitness.begin(), hivesFitness.end());	
				for(int i=0;i<int(_qtyHives/10);i++)
				{
					datafile << "Kill " << hivesFitness[i].second  << "\n" << std::endl;
					float x = ((rand()%2000)/1000.f-1.0)*border;
					float y = ((rand()%2000)/1000.f-1.0)*border;

					double* gene = new double[4];
					gene[0] = rand()%100000000/100000000.f;//0.00005;//
					gene[1] = rand()%100000000/100000000.f;//0.3;//
					gene[2] = rand()%100000000/100000000.f;//0.0001;//
					gene[3] = rand()%100000000/100000000.f;//0;//

					_hives[hivesFitness[i].second]->reset(x, y, gene);
				}
			}
		}

	}
}
