#include "environment.h"

Environment::Environment():
	_qtyHives(1), _qtyNestBoxes(20), _generation(0)
{
	// Avoid spawning on corners
	float border = 0.9;

	_nestBoxes = new NestBox[_qtyNestBoxes];

	for(int i=0; i<_qtyNestBoxes; i++)
	{
		float x = 2*((rand()%1000)/1000.f-0.5)*border;
		float y = 2*((rand()%1000)/1000.f-0.5)*border;
		float goodness = (rand()%1000)/1000.f;

		_nestBoxes[i] = NestBox(x, y, goodness);
	}

	for(int i=0; i<_qtyHives; i++)
	{
		float x = ((rand()%1000)/1000.f-0.5)*border;
		float y = ((rand()%1000)/1000.f-0.5)*border;

		float* gene = new float[4];
		gene[0] = 0.00005;//rand()%100/100000.f;
		gene[1] = 0.003;//rand()%100/100000.f;
		gene[2] = 0.001;//rand()%100/100000.f;
		gene[3] = 0;//rand()%100/100000.f;
		
		Hive* hive = new Hive(x, y, gene);
		hive->setNestBoxes(_nestBoxes, _qtyNestBoxes);
		
		_hives.push_back(hive);
	}
}

Environment::~Environment()
{
	for(Hive* hive : _hives)
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

	for(Hive* hive : _hives)
		hive->draw();
}

void Environment::plot()
{
	//float ratio = float(PLOT_WINDOW_WIDTH)/PLOT_WINDOW_HEIGHT;
	float offset = 2.0f/(_qtyNestBoxes+1);
	float size = (2.0f/(_qtyNestBoxes))*0.4f;

	int* consensus = _hives[0]->getConsensus();
	int qtyScoutBees = _hives[0]->getQtyScoutBees();

	// Plot nest boxes
	for(int i=0; i<_qtyNestBoxes; i++)
	{
		float goodness = _nestBoxes[i].getRealGoodness();
		glColor3f(goodness, 0, goodness);
		glBegin(GL_POLYGON);
		{
			glVertex2d(-1+offset*(i+1)-size, -0.9);
			glVertex2d(-1+offset*(i+1)-size, -1);
			glVertex2d(-1+offset*(i+1)+size, -1);
			glVertex2d(-1+offset*(i+1)+size, -0.9);
		}
		glEnd();

		glColor3f(0, 0, 0);
		glBegin(GL_POLYGON);
		{
			glVertex2d(-1+offset*(i+1)-size*0.5, -0.9+float(consensus[i])/qtyScoutBees);
			glVertex2d(-1+offset*(i+1)-size*0.5, -0.9);
			glVertex2d(-1+offset*(i+1)+size*0.5, -0.9);
			glVertex2d(-1+offset*(i+1)+size*0.5, -0.9+float(consensus[i])/qtyScoutBees);
		}
		glEnd();
	}

}

void Environment::run()
{
	_generation++;

	//if(_generation%100 == 0)
	//{
	//	std::cout << "Generation " << _generation << " finished!" << std::endl;
	//	int bestIndex = 0;
	//	float bestFitness = _hives[0]->getFitness();
	//	for(int i=0;i<_qtyHives;i++)
	//	{
	//		float fitness = _hives[i]->getFitness()*100.f;
	//		std::cout << "\t (" << i << ") fitness=" << fitness << std::endl;
	//		if(fitness > bestFitness)
	//		{
	//			bestIndex = i;
	//			bestFitness = fitness;
	//		}
	//	}
	//
	//	float* bestGene = _hives[bestIndex]->getGene();
	//	
	//	for(int i=0;i<_qtyHives;i++)
	//	{
	//		float* gene = _hives[i]->getGene();

	//		for(int j=0;j<4;j++)
	//			gene[j] = bestGene[j]*0.5f + gene[j]*0.5f + (rand()%1000/100000 - 500/100000);

	//		delete _hives[i];

	//		float border = 0.9;
	//		float x = ((rand()%1000)/1000.f-0.5)*border;
	//		float y = ((rand()%1000)/1000.f-0.5)*border;

	//		_hives[i] = new Hive(x, y, gene);
	//		_hives[i]->setNestBoxes(_nestBoxes, _qtyNestBoxes);
	//	}
	//}

	for(Hive* hive : _hives)
	{
		hive->run();
	}

}
