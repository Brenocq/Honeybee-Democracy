//não sei real se ta certo
class ScoutBee {       // The class
  public:             // Access specifier
    float randomChance;//chance de busca randômica
    float followChance;//Chance de seguir outra q ta perto
    float linearDecay; //constante linear de decaimento (0-1)
    float quadraDecay; //constante quadrática de decaimento (0-1)
    float consensus;
  private:
    int posx, posy;
    float theta;
    float choose;
    float speed;
};
