#include <iostream>
#include <iterator>

#include "algo/atrous.h"
#include "algo/simplepicker.h"
#include "algo/projectionstats.h"



void testProjectionStats(){

   float data[] = {30.};


  std::vector<float> project(data, data + sizeof(data)/sizeof(float));
  {
    ralab::projectionstats p(project,0);
    std::cout << p.getApex() << " " << p.getCenterOfMass() << " " << p.getKurtosis() << " "
              << p.getSD() << " " << p.getSkewness() << std::endl;
  }


  float data2[] = {225. , 393.768, 460.217,
                   271.681, 134.084, 53.0417};

  std::vector<float> projection(data2, data2 + sizeof(data2)/sizeof(float));
  {
    ralab::projectionstats p(projection,0);
    std::cout << p.getApex() << " " << p.getCenterOfMass() << " " << p.getKurtosis() << " "
              << p.getSD() << " " << p.getSkewness() << std::endl;
  }

  {
    projection.pop_back();
    ralab::projectionstats p(projection,0);
    std::cout << projection.size() << std::endl;
    std::cout << p.getApex() << " " << p.getCenterOfMass() << " " << p.getKurtosis() << " "
              << p.getSD() << " " << p.getSkewness() << std::endl;
  }

  {
    projection.pop_back();
    ralab::projectionstats p(projection,0);
    std::cout << projection.size() << std::endl;
    std::cout << p.getApex() << " " << p.getCenterOfMass() << " " << p.getKurtosis() << " "
              << p.getSD() << " " << p.getSkewness() << std::endl;
  }

  {
    projection.pop_back();
    ralab::projectionstats p(projection,0);
    std::cout << projection.size() << std::endl;
    std::cout << p.getApex() << " " << p.getCenterOfMass() << " " << p.getKurtosis() << " "
              << p.getSD() << " " << p.getSkewness() << std::endl;
  }

  {
    projection.pop_back();
    ralab::projectionstats p(projection,0);
    std::cout << projection.size() << std::endl;
    std::cout << p.getApex() << " " << p.getCenterOfMass() << " " << p.getKurtosis() << " "
              << p.getSD() << " " << p.getSkewness() << std::endl;
  }
}


void simplePickerTest(){
  float data[] = {200., 393.768,
                  460.217, 271.681,
                  134.084, 53.0417};
  float x = ralab::simplePicker(data , data+sizeof(data)/sizeof(float), 0. );
  if(x < 0 ){
      std::cout << "OK because d 0" << std::endl;
    }

  float y = ralab::simplePicker(data , data+sizeof(data)/sizeof(float), 1. );
  if(y > 0){
      std::cout << "because peak analysed" << std::endl;
    }

  std::cout << y << std::endl;
  float data2[] = {393.768, 460.217,
                   271.681, 134.084, 53.0417};

  x = ralab::simplePicker(data2 , data2+sizeof(data2)/sizeof(float), 1. );
  if(y < 0){
      std::cout << x << " OK because not enough datapoints" << std::endl;
    }
}



int main(int arg, char ** argv)
{
  simplePickerTest();
  testProjectionStats();
  typedef vigra::ATrous<float> mATrous;
  mATrous atrous(6);
  mATrous::Kernel kernel0, kernel1, kernel2;

  atrous.read("../data/napedro_L120420_009_SWfitered.tiff");
  atrous.write_image(0,"../data/human2","orig");
  atrous.convolveImageStack(1.);
  atrous.computeDifferences();
  atrous.write_images("../data/human2", "decomposed");

  atrous.init(5,0.);
  atrous.init(0,0.);
  atrous.write_images("../data/human2", "filtered");
  atrous.synthesis();
  atrous.write_image(0,"../data/human2","synth");


  std::ostream_iterator<float> out_it (std::cout,", ");
  atrous.initGaussKernel(1.,0,kernel0);
  atrous.initGaussKernel(1,1,kernel1);
  atrous.initGaussKernel(1,2,kernel2);

  std::copy(kernel0.begin(), kernel0.end(), out_it);
  std::cout << std::endl;
  std::copy(kernel1.begin(), kernel1.end(), out_it);
  std::cout << std::endl;
  std::copy(kernel2.begin(), kernel2.end(), out_it);
  std::cout << std::endl;


  //int res = mATrous::powI(2,3);
  return 0;
}
