#############################################
#load MegaDetector tensorflow model from .pb file
#' Title
#'
#' @param modelfile .pb file obtained from megaDetector
#'
#' @return a tfsession containing the MD model
#' @export
#' @import tensorflow
#'
#' @examples
#' \dontrun{
#' mdmodel <- "megadetector_v4.1.pb"
#' mdsession <- loadMDModel(mdmodel)
#' }
loadMDModel <- function(modelfile) {
  require(tensorflow)
  if (!file.exists(modelfile)) {
    stop("The given MD model does not exist. Check path.")
  }
  tfsession <- tf$compat$v1$Session()
  f <- tf$io$gfile$GFile(modelfile, "rb")
  tfgraphdef <- tf$compat$v1$GraphDef()
  tfgraphdef$ParseFromString(f$read())
  tfsession$graph$as_default()
  tf$import_graph_def(tfgraphdef, name = "")
  class(tfsession) <- append(class(tfsession), "mdsession")
  tfsession
}