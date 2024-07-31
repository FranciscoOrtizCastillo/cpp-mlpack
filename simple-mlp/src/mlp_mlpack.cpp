#include <mlpack/mlpack.hpp>

int main(){
  
  arma::mat data;

  //mlpack::data::Load("data/foo.csv",data,true);
  if (!mlpack::data::Load("data/foo.csv", data))
    throw std::runtime_error("Could not read data/foo.csv!");

  //data.print();

  arma::mat trainData = data.submat(0,0,data.n_rows-2,data.n_cols-6);
  arma::mat trainLabels = data.submat(data.n_rows-1,0,data.n_rows-1,data.n_cols-6);

  //trainData.print();
  //trainLabels.print();

  arma::mat testData = data.submat(0,data.n_cols-5,data.n_rows-2,data.n_cols-1);
  arma::mat testLabels = data.submat(data.n_rows-1,data.n_cols-5,data.n_rows-1,data.n_cols-1);

  //std::cout << trainData << std::endl;
  //std::cout << trainLabels << std::endl;

  // Initialize the network.
  mlpack::FFN<mlpack::ann::MeanSquaredError, mlpack::RandomInitialization> model;

  model.Add<mlpack::Linear>(8);
  model.Add<mlpack::Sigmoid>();
  model.Add<mlpack::Linear>(8);
  model.Add<mlpack::Sigmoid>();
  model.Add<mlpack::Linear>(1);
  model.Add<mlpack::Sigmoid>();
  
  for (int i = 0; i < 4; ++i)
  {
    //std::cout << "Train " << i << std::endl;

    model.Train(trainData, trainLabels);
  }
  
  arma::mat prediction;

  model.Predict(testData, prediction);

  std::cout << "Predictions    : " << prediction << std::endl;
  std::cout << "Correct Labels : " << testLabels << std::endl;

  // mlpack::data::Save("cov.csv",cov,true);
  return 0;
}