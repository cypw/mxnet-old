/*!
 *  Copyright (c) 2015 by Contributors
 * \file image_aug_torch.cc
 * \brief Torch augmenter.
 */
#include <mxnet/base.h>
#include <utility>
#include <string>
#include <algorithm>
#include <vector>
#include "./image_augmenter.h"
#include "../common/utils.h"

#if MXNET_USE_OPENCV
// Registers
namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::io::ImageAugmenterReg);
}  // namespace dmlc
#endif

namespace mxnet {
namespace io {

/*! \brief image augmentation parameters*/
struct TorchImageAugmentParam : public dmlc::Parameter<TorchImageAugmentParam> {
  /*! \brief whether we do random cropping */
  bool rand_crop;
  /*! \brief whether we do nonrandom croping */
  int crop_y_start;
  /*! \brief whether we do nonrandom croping */
  int crop_x_start;
  /*! \brief [-max_rotate_angle, max_rotate_angle] */
  int max_rotate_angle;
  /*! \brief min aspect ratio */
  float min_aspect_ratio;
  /*! \brief max aspect ratio */
  float max_aspect_ratio;
  /*! \brief random shear the image [-max_shear_ratio, max_shear_ratio] */
  float max_shear_ratio;
  /*! \brief max crop size */
  int max_crop_size;
  /*! \brief min crop size */
  int min_crop_size;
  /*! \brief max area ratio */
  float max_random_area;
  /*! \brief min area_ratio */
  float min_random_area;
  /*! \brief max scale ratio */
  float max_random_scale;
  /*! \brief min scale_ratio */
  float min_random_scale;
  /*! \brief min image size */
  float min_img_size;
  /*! \brief max image size */
  float max_img_size;
  /*! \brief max random in H channel */
  int random_h;
  /*! \brief max random in S channel */
  int random_s;
  /*! \brief max random in L channel */
  int random_l;
  /*! \brief std value for PCA lighting */
  float lighting_std;
  /*! \brief rotate angle */
  int rotate;
  /*! \brief filled color while padding */
  TShape fill_value;
  /*! \brief interpolation method 0-NN 1-bilinear 2-cubic 3-area 4-lanczos4 9-auto 10-rand  */
  int inter_method;
  /*! \brief padding size */
  int pad;
  /*! \brief shape of the image data*/
  TShape data_shape;
  // declare parameters
  DMLC_DECLARE_PARAMETER(TorchImageAugmentParam) {
    DMLC_DECLARE_FIELD(rand_crop).set_default(false)
        .describe("Augmentation Param: Whether to random crop on the image");
    DMLC_DECLARE_FIELD(crop_y_start).set_default(-1)
        .describe("Augmentation Param: Where to nonrandom crop on y.");
    DMLC_DECLARE_FIELD(crop_x_start).set_default(-1)
        .describe("Augmentation Param: Where to nonrandom crop on x.");
    DMLC_DECLARE_FIELD(max_rotate_angle).set_default(0.0f)
        .describe("Augmentation Param: rotated randomly in [-max_rotate_angle, max_rotate_angle].");
    DMLC_DECLARE_FIELD(min_aspect_ratio).set_default(1.0f)
        .describe("Augmentation Param: denotes the min ratio of random aspect ratio augmentation.");
    DMLC_DECLARE_FIELD(max_aspect_ratio).set_default(1.0f)
        .describe("Augmentation Param: denotes the max ratio of random aspect ratio augmentation.");
    DMLC_DECLARE_FIELD(max_shear_ratio).set_default(0.0f)
        .describe("Augmentation Param: denotes the max random shearing ratio.");
    DMLC_DECLARE_FIELD(max_crop_size).set_default(-1)
        .describe("Augmentation Param: Maximum crop size.");
    DMLC_DECLARE_FIELD(min_crop_size).set_default(-1)
        .describe("Augmentation Param: Minimum crop size.");
    DMLC_DECLARE_FIELD(max_random_area).set_default(1.0f)
        .describe("Augmentation Param: Maxmum area ratio.");
    DMLC_DECLARE_FIELD(min_random_area).set_default(1.0f)
        .describe("Augmentation Param: Minimum area ratio.");
    DMLC_DECLARE_FIELD(max_random_scale).set_default(1.0f)
        .describe("Augmentation Param: Maxmum scale ratio.");
    DMLC_DECLARE_FIELD(min_random_scale).set_default(1.0f)
        .describe("Augmentation Param: Minimum scale ratio.");
    DMLC_DECLARE_FIELD(max_img_size).set_default(1e10f)
        .describe("Augmentation Param: Maxmum image size after resizing.");
    DMLC_DECLARE_FIELD(min_img_size).set_default(0.0f)
        .describe("Augmentation Param: Minimum image size after resizing.");
    DMLC_DECLARE_FIELD(random_h).set_default(0)
        .describe("Augmentation Param: Maximum value of H channel in HSL color space.");
    DMLC_DECLARE_FIELD(random_s).set_default(0)
        .describe("Augmentation Param: Maximum value of S channel in HSL color space.");
    DMLC_DECLARE_FIELD(random_l).set_default(0)
        .describe("Augmentation Param: Maximum value of L channel in HSL color space.");
    DMLC_DECLARE_FIELD(lighting_std).set_default(0)
        .describe("Augmentation Param: std value for PCA lighting");
    DMLC_DECLARE_FIELD(rotate).set_default(-1.0f)
        .describe("Augmentation Param: Rotate angle.");
    DMLC_DECLARE_FIELD(fill_value)
        .set_expect_ndim(3)
        .describe("Augmentation Param: Fill border value(rgb).");
    DMLC_DECLARE_FIELD(data_shape)
        .set_expect_ndim(3).enforce_nonzero()
        .describe("Dataset Param: Shape of each instance generated by the DataIter.");
    DMLC_DECLARE_FIELD(inter_method).set_default(1)
        .describe("Augmentation Param: 0-NN 1-bilinear 2-cubic 3-area 4-lanczos4 9-auto 10-rand.");
    DMLC_DECLARE_FIELD(pad).set_default(0)
        .describe("Augmentation Param: Padding size.");
  }
};

DMLC_REGISTER_PARAMETER(TorchImageAugmentParam);

std::vector<dmlc::ParamFieldInfo> ListTorchAugParams() {
  return TorchImageAugmentParam::__FIELDS__();
}

#if MXNET_USE_OPENCV

#ifdef _MSC_VER
#define M_PI CV_PI
#endif
/*! \brief helper class to do image augmentation */
class TorchImageAugmenter : public ImageAugmenter {
 public:
  // contructor
  TorchImageAugmenter() {
    rotateM_ = cv::Mat(2, 3, CV_32F);
  }
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    kwargs_left = param_.InitAllowUnknown(kwargs);
    for (size_t i = 0; i < kwargs_left.size(); i++) {
        if (!strcmp(kwargs_left[i].first.c_str(), "rotate_list")) {
          const char* val = kwargs_left[i].second.c_str();
          const char *end = val + strlen(val);
          char buf[128];
          while (val < end) {
            sscanf(val, "%[^,]", buf);
            val += strlen(buf) + 1;
            rotate_list_.push_back(atoi(buf));
          }
        }
    }
  }
  /*!
   * \brief get interpolation method with given inter_method, 0-CV_INTER_NN 1-CV_INTER_LINEAR 2-CV_INTER_CUBIC
   * \ 3-CV_INTER_AREA 4-CV_INTER_LANCZOS4 9-AUTO(cubic for enlarge, area for shrink, bilinear for others) 10-RAND
   */
  int GetInterMethod(int inter_method, int old_width, int old_height, int new_width,
    int new_height, common::RANDOM_ENGINE *prnd) {
    if (inter_method == 9) {
      if (new_width > old_width && new_height > old_height) {
        return 2;  // CV_INTER_CUBIC for enlarge
      } else if (new_width <old_width && new_height < old_height) {
        return 3;  // CV_INTER_AREA for shrink
      } else {
        return 1;  // CV_INTER_LINEAR for others
      }
      } else if (inter_method == 10) {
      std::uniform_int_distribution<size_t> rand_uniform_int(0, 4);
      return rand_uniform_int(*prnd);
    } else {
      return inter_method;
    }
  }
  /*!
   * \brief Given a rectangle of size wxh that has been rotated by 'angle' (in
   * radians), computes the width and height of the largest possible
   * axis-aligned rectangle within the rotated rectangle.

   * Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
   */
  cv::Size largest_rotated_rect(int src_h, int src_w, int angle){
    CHECK(src_h > 0 && src_w > 0);
    int wr, hr;
    float side_short = std::min(src_h, src_w);
    float side_long  = std::max(src_h, src_w);
    float sin_a = fabs(sin(float(angle) / 180.0 * M_PI));
    float cos_a = fabs(cos(float(angle) / 180.0 * M_PI));
    if(side_short <= 2*sin_a*cos_a*side_long) {
      float x = 0.5 * side_short;
      if (src_w > src_h) {
        wr = x/sin_a;
        hr = x/cos_a;
      } else {
        wr = x/cos_a;
        hr = x/sin_a;
      }
    } else {
      float cos_2a = cos_a*cos_a - sin_a*sin_a;
      wr = (src_w*cos_a - src_h*sin_a)/cos_2a;
      hr = (src_h*cos_a - src_w*sin_a)/cos_2a;
    }
    return cv::Size(wr, hr);
  }
  cv::Mat Process(const cv::Mat &src,
                  common::RANDOM_ENGINE *prnd) override {
    using mshadow::index_t;
    cv::Mat res = src.clone();
    CHECK((param_.inter_method >= 1 && param_.inter_method <= 4) ||
          (param_.inter_method >= 9 && param_.inter_method <= 10))
           << "invalid inter_method: valid value 0,1,2,3,9,10";

   // do aspect ratio before affine transformation to reduce black border
   if (param_.max_aspect_ratio != 1.0 || param_.min_aspect_ratio != 1.0) {
      CHECK(param_.min_aspect_ratio <= param_.max_aspect_ratio);
      std::uniform_real_distribution<float> rand_uniform(0, 1);
      float ratio = 0.5 * ( rand_uniform(*prnd) + rand_uniform(*prnd) ) *
          (param_.max_aspect_ratio - param_.min_aspect_ratio) + param_.min_aspect_ratio;
      float w_ratio = 1/ratio;
      float h_ratio = 1*ratio;
      if (rand_uniform(*prnd) > 0.5) {
          float tmp_ratio = w_ratio;
          w_ratio = h_ratio;
          h_ratio = tmp_ratio;
      }
      // we enlarge the original size to avoid downsampling
      int new_width  = std::ceil(res.cols * w_ratio * 1.1);
      int new_height = std::ceil(res.rows * h_ratio * 1.1);
      int interpolation_method = GetInterMethod(param_.inter_method,
                     res.cols, res.rows, new_width, new_height, prnd);
      cv::resize(res, temp_, cv::Size(new_width, new_height), 0, 0, interpolation_method);
      res = temp_;
    }

    { // always resize input image according to crop size, since no pre-processing is required.
      std::uniform_real_distribution<float> rand_uniform(0, 1);
      // shear
      float s = rand_uniform(*prnd) * param_.max_shear_ratio * 2 - param_.max_shear_ratio;
      // rotate
      int angle = 0.5 * (std::uniform_int_distribution<int>(-param_.max_rotate_angle, param_.max_rotate_angle)(*prnd)
                        +std::uniform_int_distribution<int>(-param_.max_rotate_angle, param_.max_rotate_angle)(*prnd));
      if (param_.rotate > 0) angle = param_.rotate;
      if (rotate_list_.size() > 0) {
        angle = rotate_list_[std::uniform_int_distribution<int>(0, rotate_list_.size() - 1)(*prnd)];
      }
      float a = cos(angle / 180.0 * M_PI);
      float b = sin(angle / 180.0 * M_PI);
      // scale, +0.4 to avoid rounding problem
      float scale_norm = std::max((0.4 + (float)param_.data_shape[1])/(float)res.rows,
                                  (0.4 + (float)param_.data_shape[2])/(float)res.cols);
      CHECK( (param_.max_random_area == 1.0 && param_.min_random_area == 1.0)
             || (param_.max_random_scale == 1.0 && param_.min_random_scale == 1.0) );
      float scale;
      if (param_.max_random_area != 1.0 || param_.min_random_area != 1.0) {
          // random area
          float rnd_area  = rand_uniform(*prnd) *
              (param_.max_random_area - param_.min_random_area) + param_.min_random_area;
          scale = scale_norm / std::sqrt(rnd_area);
      } else {
          // random scale
          float rnd_scale = rand_uniform(*prnd) *
              (param_.max_random_scale - param_.min_random_scale) + param_.min_random_scale;
          scale = scale_norm * rnd_scale;
      }
      // aspect ratio is finished before affine transformation 
      float hs = scale;
      float ws = hs;
      // new width and height
      float new_width = std::max(param_.min_img_size,
                                 std::min(param_.max_img_size, scale * res.cols));
      float new_height = std::max(param_.min_img_size,
                                 std::min(param_.max_img_size, scale * res.rows));
      cv::Mat M(2, 3, CV_32F);
      M.at<float>(0, 0) = hs * a - s * b * ws;
      M.at<float>(1, 0) = -b * ws;
      M.at<float>(0, 1) = hs * b + s * a * ws;
      M.at<float>(1, 1) = a * ws;
      float ori_center_width = M.at<float>(0, 0) * res.cols + M.at<float>(0, 1) * res.rows;
      float ori_center_height = M.at<float>(1, 0) * res.cols + M.at<float>(1, 1) * res.rows;
      M.at<float>(0, 2) = (new_width - ori_center_width) / 2;
      M.at<float>(1, 2) = (new_height - ori_center_height) / 2;
      int interpolation_method = GetInterMethod(param_.inter_method,
                     res.cols, res.rows, new_width, new_height, prnd);
      cv::warpAffine(res, temp_, M, cv::Size(new_width, new_height),
                     interpolation_method,
                     cv::BORDER_CONSTANT,
                     cv::Scalar(param_.fill_value[2], param_.fill_value[1], param_.fill_value[0]));
      if (angle != 0) {
        cv::Size crop_wh = largest_rotated_rect(new_height, new_width, angle);
        index_t x = (temp_.cols - crop_wh.width)/2;
        index_t y = (temp_.rows - crop_wh.height)/2;
        cv::Rect roi(x, y, crop_wh.width, crop_wh.height);
        float scale_kw_ = std::max(1.f, ((float)param_.data_shape[2]/crop_wh.width));
        float scale_kh_ = std::max(1.f, ((float)param_.data_shape[1]/crop_wh.height));
        float scale_kx_ = std::max(scale_kw_, scale_kh_);
        new_width  = std::ceil(scale_kx_ * crop_wh.width);
        new_height = std::ceil(scale_kx_ * crop_wh.height);
        int interpolation_method = GetInterMethod(param_.inter_method, crop_wh.width, crop_wh.height,
                                                  new_width, new_height, prnd);
        cv::resize(temp_(roi), res, cv::Size(new_width, new_height), 0, 0, interpolation_method);
      } else {
        res = temp_;
      }
    }

    // pad logic
    if (param_.pad > 0) {
      cv::copyMakeBorder(res, res, param_.pad, param_.pad, param_.pad, param_.pad,
                         cv::BORDER_CONSTANT,
                         cv::Scalar(param_.fill_value[2], param_.fill_value[1], param_.fill_value[0]));
    }

    // crop logic
    if (param_.max_crop_size != -1 || param_.min_crop_size != -1) {
      CHECK(res.cols >= param_.max_crop_size && res.rows >= \
              param_.max_crop_size && param_.max_crop_size >= param_.min_crop_size)
          << "input image size smaller than max_crop_size";
      index_t rand_crop_size =
          std::uniform_int_distribution<index_t>(param_.min_crop_size, param_.max_crop_size)(*prnd);
      index_t y = res.rows - rand_crop_size;
      index_t x = res.cols - rand_crop_size;
      if (param_.rand_crop != 0) {
        y = std::uniform_int_distribution<index_t>(0, y)(*prnd);
        x = std::uniform_int_distribution<index_t>(0, x)(*prnd);
      } else {
        y /= 2; x /= 2;
      }
      cv::Rect roi(x, y, rand_crop_size, rand_crop_size);
      int interpolation_method = GetInterMethod(param_.inter_method, rand_crop_size, rand_crop_size,
                                                param_.data_shape[2], param_.data_shape[1], prnd);
      cv::resize(res(roi), res, cv::Size(param_.data_shape[2], param_.data_shape[1])
                , 0, 0, interpolation_method);
    } else {
      CHECK(static_cast<index_t>(res.rows) >= param_.data_shape[1]
            && static_cast<index_t>(res.cols) >= param_.data_shape[2])
          << "input image size smaller than input shape";
      index_t y = res.rows - param_.data_shape[1];
      index_t x = res.cols - param_.data_shape[2];
      if (param_.rand_crop != 0) {
        y = std::uniform_int_distribution<index_t>(0, y)(*prnd);
        x = std::uniform_int_distribution<index_t>(0, x)(*prnd);
      } else {
        y /= 2; x /= 2;
      }
      cv::Rect roi(x, y, param_.data_shape[2], param_.data_shape[1]);
      res = res(roi);
    }

   // TODO: Context-aware Lighting Augmentation
   if (param_.lighting_std > 0) { 
      // for the consideration of image segmenation, we put rand out of the scope
      std::normal_distribution<float> rand_normal(1, param_.lighting_std);
      float rand_noise[9];
      for (int i = 0; i < 9; i++)
        rand_noise[i] = rand_normal(*prnd);
      if (res.channels() == 3) {
        res.convertTo(res, CV_32FC3);
        cv::Mat data = res.reshape(1, res.rows * res.cols);
        cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, 3);
        cv::Mat points = pca.project(data);
        cv::Mat eigenvectors = pca.eigenvectors;
        // add gaussian noise
        for (int i_n = 0; i_n < 3; i_n++) {
          for (int i_dim = 0; i_dim < 3; i_dim++) {
            eigenvectors.at<float>(i_n,i_dim) = rand_noise[3*i_n+i_dim] * eigenvectors.at<float>(i_n,i_dim);
          }
        }
        temp_ = pca.backProject(points);
        temp_ = temp_.reshape(res.channels(), res.rows);
        temp_.convertTo(res, CV_8UC3);
      }
   }

    // color space augmentation
    if (param_.random_h != 0 || param_.random_s != 0 || param_.random_l != 0) {
      std::uniform_real_distribution<float> rand_uniform(0, 1);
      cvtColor(res, res, CV_BGR2HLS);
      // use an approximation of gaussian distribution to reduce extreme value
      float rh = rand_uniform(*prnd); rh += 4 * rand_uniform(*prnd); rh = rh / 5;
      float rs = rand_uniform(*prnd); rs += 4 * rand_uniform(*prnd); rs = rs / 5;
      float rl = rand_uniform(*prnd); rl += 4 * rand_uniform(*prnd); rl = rl / 5;
      int h = rh * param_.random_h * 2 - param_.random_h;
      int s = rs * param_.random_s * 2 - param_.random_s;
      int l = rl * param_.random_l * 2 - param_.random_l;
      int temp[3] = {h, l, s};
      int limit[3] = {180, 255, 255};
      for (int i = 0; i < res.rows; ++i) {
        for (int j = 0; j < res.cols; ++j) {
          for (int k = 0; k < 3; ++k) {
            int v = res.at<cv::Vec3b>(i, j)[k];
            v += temp[k];
            v = std::max(0, std::min(limit[k], v));
            res.at<cv::Vec3b>(i, j)[k] = v;
          }
        }
      }
      cvtColor(res, res, CV_HLS2BGR);
    }
    return res;
  }

 private:
  // temporal space
  cv::Mat temp_;
  // rotation param
  cv::Mat rotateM_;
  // parameters
  TorchImageAugmentParam param_;
  /*! \brief list of possible rotate angle */
  std::vector<int> rotate_list_;
};

ImageAugmenter* ImageAugmenter::Create(const std::string& name) {
  return dmlc::Registry<ImageAugmenterReg>::Find(name)->body();
}

MXNET_REGISTER_IMAGE_AUGMENTER(aug_torch)
.describe("torch augmenter")
.set_body([]() {
    return new TorchImageAugmenter();
  });
#endif  // MXNET_USE_OPENCV
}  // namespace io
}  // namespace mxnet
