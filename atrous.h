#ifndef ATROUS_H
#define ATROUS_H
#include <vigra/multi_array.hxx>
#include <vigra/convolution.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/imagecontainer.hxx>
#include <vector>
#include <boost/filesystem.hpp>
#include <vigra/impex.hxx>


#include "stats/dwt/dshrinkage.h"

namespace vigra{
  namespace fs = boost::filesystem;

  template <class TValue >
  class ATrous
  {
  public:
    typedef MultiArray<2, TValue> ImageType;
    typedef MultiArray<1, TValue> Kernel;
    typedef typename ImageType::difference_type difference_type;

  private:
    std::size_t scale_; // scale of decomposition
    difference_type size_; // size of image
    std::vector<ImageType> images_;
    std::vector<Kernel> kernels_;


  public:
    ATrous(std::size_t scale):
      scale_(scale),
      size_(0,0),
      images_(scale)
    {}

    std::size_t getScale(){
      return scale_;
    }

    void reshape(const difference_type & size){
      for(int i = 0 ; i < images_.size(); ++i){
          images_[i].reshape(size_);
        }
    }


    // set image
    void setImage(ImageType & image){
      images_[0] = image;
      this->size_ =difference_type(image.size(0), image.size(1));
      for(int i = 1 ; i < images_.size(); ++i){
          images_[i].reshape(size_);
        }
    }

    ImageType & getImage(int i ){
      return images_[i];
    }


  private:
    static int powI(int val , int exp){
      if(exp == 0)
        return 1;
      int res = val;
      exp -=1;
      for(int i = 0; i < exp; ++i){
          res = res * val;
        }
      return res;
    }

  public:
    //initialize the Gauss kernel
    void initGaussKernel(double sigma,int scale, Kernel & kernel ){
      int size = static_cast<int>(3*sigma);
      int sizeX = 2*size + 1;
      int spacer = powI(2,scale);
      int size2 = (sizeX-1)*spacer + 1;
      kernel.reshape(typename Kernel::difference_type(size2));
      Gaussian<TValue> gauss(sigma);
      for(int y=0; y<sizeX; ++y)
        {
          kernel(spacer * y) = gauss( static_cast<TValue>(y) - static_cast<TValue>(size) ) ;
        }
    }

    void initKernel1D(Kernel & kernel,vigra::Kernel1D<float> & kx)
    {
      int xsize = kernel.size();
      int xsizeHalf = (xsize-1)/2;
      kx.initExplicitly(-xsizeHalf,xsizeHalf) = 0.;
      int j = 0;
      for(int i = -xsizeHalf; i<=xsizeHalf; ++i, ++j){
          kx[i] = kernel[j];
        }
    }

    /// convolve image
    void convolveAtScale(TValue sigma, int scale, const ImageType & orig, ImageType & res){
      Kernel kernel;
      //init the kernels
      initGaussKernel(sigma,scale, kernel);

      std::vector<vigra::Kernel1D<float> > kx(2) ;
      initKernel1D(kernel,kx[0]);
      initKernel1D(kernel,kx[1]);
      vigra::separableConvolveMultiArray(srcMultiArrayRange(orig), destMultiArray(res),
                                         kx.begin());
    }

    /// Convolves all images in the stack.
    void convolveImageStack(TValue sigma){
      for(std::size_t i = 1 ; i < images_.size();++i)
        {
          convolveAtScale(sigma,i-1,images_[i-1],images_[i]);
        }
    }

    // compute the differences
    void computeDifferences(){
      for(std::size_t i = 1 ; i < images_.size();++i)
        {
          images_[i-1] -= images_[i];
        }
    }


    void shrinkHard(ImageType & gradient , TValue threshold){
      ralab::STATS::HardShrinkage<TValue> hs(threshold);
      auto beg = gradient.data();
      auto end = beg + gradient.size();
      std::transform(beg, end, beg, hs);
    }

    void shrinkSoft(ImageType & gradient , TValue threshold){
      ralab::STATS::SoftShrinkage<TValue> hs(threshold);
      auto beg = gradient.data();
      auto end = beg + gradient.size();
      std::transform(beg, end, beg, hs);
    }


    void shrinkHardDetails(TValue threshold){
      for(int i = 0 ; i < (images_.size()-1);++i)
        {
          shrinkHard(images_[i], threshold);
        }
    }

    void shrinkSoftDetails(TValue threshold){
      for(int i = 0 ; i < images_.size()-1 ;++i)
        {
          shrinkSoft(images_[i], threshold);
        }
    }


    void shrinkSoftApproximate(TValue threshold){
      shrinkSoft(images_.back(), threshold);
    }


    //init image
    void init(int i, TValue val){
      images_[i].init(val);
    }

    // synthesis operation
    void synthesis(){
      for(std::size_t i = 1 ; i < images_.size();++i)
        {
          images_[0] += images_[i];
        }
    }

    // synthesis operation
    void synthesis(TValue amp){
      using namespace vigra::multi_math;
      for(int i = images_.size() -1 ; i >=0;--i)
        {
          images_.back() += amp * images_[i];
        }
      images_[0] = images_.back();
    }

    struct RepNeg{
      TValue x_;
      RepNeg(TValue x):x_(x){}
      TValue operator()(TValue val){
        if(val < x_)
          return 0;
        else
          return val;
      }
    };

    void remNegative(ImageType & gradient,TValue thresh){
      auto beg = gradient.data();
      auto end = beg + gradient.size();
      std::transform(beg, end, beg, RepNeg(thresh));
    }


    void remNegative(int i, TValue thresh){
      remNegative(images_[i],thresh);
    }


    //read initialization image
    void read(const std::string & file){
      boost::filesystem::path p1(file);
      if(boost::filesystem::exists(p1)){
          vigra::ImageImportInfo info(p1.string().c_str());
          if(info.isGrayscale())
            {
              images_[0].reshape(difference_type(info.width(), info.height()));
              vigra::importImage(info, destImage(images_[0]));
            }
          this->size_ =difference_type(info.width(), info.height());
          for(std::size_t i = 1 ; i < images_.size(); ++i){
              images_[i].reshape(size_);
            }
          //this->reshape(difference_type(info.width(), info.height()));
        }
    }


    void write_image(int i, const std::string & foldername, const std::string & desc = "" ){
      fs::path fn(foldername);
      fs::path ffull;
      ffull = fs::absolute(fn);
      if(!is_directory(ffull))
        {
          boost::filesystem::create_directories(ffull);
        }
      std::stringstream sstr;
      sstr << desc  << i  << ".png";
      fs::path x = ffull;
      x /= sstr.str();
      vigra::ImageExportInfo iei(x.string().c_str());
      vigra::exportImage(vigra::srcImageRange(images_[i]),iei);
    }

    //writes all images in the stack into a folder
    void write_images(const std::string & foldername, const std::string & desc = ""){
      using namespace boost::filesystem;
      path fn(foldername);
      path ffull;
      ffull = fs::absolute(fn);
      if(!is_directory(ffull))
        {
          boost::filesystem::create_directories(ffull);
        }

      for(std::size_t i = 0 ; i < images_.size(); ++i){
          std::stringstream sstr ;
          sstr << desc << i << ".tiff";
          path x = ffull;
          x /= sstr.str();
          vigra::ImageExportInfo iei(x.string().c_str());
          iei.setFileType("TIFF");
          vigra::exportImage(vigra::srcImageRange(images_[i]),iei);
        }
    }

  };
}

#endif // ATROUS_H
