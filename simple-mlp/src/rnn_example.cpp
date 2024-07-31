#include <mlpack/mlpack.hpp>

using namespace ens;
using namespace mlpack;

/**
 * Generates noisy sine wave and outputs the data and the labels that
 * can be used directly for training and testing with RNN.
 */
void GenerateNoisySines(arma::cube& data,
                        arma::cube& labels,
                        size_t rho,
                        const size_t dataPoints = 100,
                        const double noisePercent = 0.2)
{
  size_t points = dataPoints;
  size_t r = dataPoints % rho;

  if (r == 0)
    points += 1;
  else
    points += rho - r + 1;

  arma::colvec x(points);
  int i = 0;
  double interval = 0.6 / points;

  x.for_each([&i, noisePercent, interval]
    (arma::colvec::elem_type& val) {
    double t = interval * (++i);
    val = ::sin(2 * M_PI * 10 * t) + (noisePercent * Random(0.0, 0.1));
  });

  arma::colvec y = x;
  y = arma::normalise(x);

  // Now break this into columns of rho size slices.
  size_t numColumns = y.n_elem / rho;
  data = arma::cube(1, numColumns, rho);
  labels = arma::cube(1, numColumns, 1);

  for (size_t i = 0; i < numColumns; ++i)
  {
    data.tube(0, i) = y.rows(i * rho, i * rho + rho - 1);
    labels.subcube(0, i, 0, 0, i, 0) =
        y.rows(i * rho + rho, i * rho + rho);
  }
}

int main()
{
  const size_t rho = 10;

  // Generate 12 (2 * 6) noisy sines. A single sine contains rho
  // points/features.
  arma::cube input, labels;
  GenerateNoisySines(input, labels, rho);

  /**
   * Construct a network with 1 input unit, 4 LSTM units and 1 output
   * unit. The hidden layer is connected to itself. The network structure
   * looks like:
   *
   *  Input         Hidden        Output
   * Layer(1)      LSTM(4)       Layer(1)
   * +-----+       +-----+       +-----+
   * |     |       |     |       |     |
   * |     +------>|     +------>|     |
   * |     |    ..>|     |       |     |
   * +-----+    .  +--+--+       +-----+
   *            .     .
   *            .     .
   *            .......
   *
   * We use MeanSquaredError for the loss type, since we are predicting a
   * continuous value.
   */
  RNN<MeanSquaredError> model(rho, true /* only one response per sequence */);
  model.Add<LSTM>(4);
  model.Add<LinearNoBias>(1);

  StandardSGD opt(0.1, 1, 10 * input.n_cols /* 10 epochs */, -100);
  model.Train(input, labels, opt);

  // Now compute the MSE on the training set.
  arma::cube predictions;
  model.Predict(input, predictions);
  const double mse = arma::accu(arma::square(
      arma::vectorise(labels) -
      arma::vectorise(predictions.slice(predictions.n_slices - 1)))) /
      input.n_cols;
  std::cout << "MSE on training set is " << mse << "." << std::endl;
}