import io
import logging
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained = True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    logger.info(f"Model output features {model.fc.in_features}")
    num_classes = 133
    model.fc = nn.Sequential(nn.Linear(num_features, 256), 
                              nn.ReLU(),
                              nn.Linear(256,  num_classes),
                              nn.LogSoftmax(dim=1)
                            )
    
    return model

# Define the model architecture - this should match the architecture of the trained model
def model_fn(model_dir):
    """
    Load the PyTorch model from the `model_dir` directory.
    """
    logger.info(f" inference >>>>> Loading model from {model_dir}")

    # Instantiate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)

    # Update the fully connected layer to match the output features (num_classes)
    # Ensure that the number of features matches the model's output during training
    #num_classes = 133 
    #model.fc = nn.Linear(model.fc.in_features, num_classes)
    #model.fc = nn.Sequential(nn.Linear(num_features, 256), 
    #                          nn.ReLU(),
    #                          nn.Linear(256,  num_classes),
    #                          nn.LogSoftmax(dim=1)
    #                        )

    

    # Load the saved model weights
    model_path = f"{model_dir}/model.pth"
    try:
        checkpoint = torch.load(model_path, map_location=device)
        #model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        #checkpoint = {k: v for k, v in checkpoint.items() if k.startswith('conv')}

        # Load the state dictionary into the model
        model.load_state_dict(checkpoint)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception(f"Error loading the model from {model_path}.")
        raise e

    # Set the model to evaluation mode
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    """
    Preprocess the input data.
    """
    logger.info("In input_fn.")
    if request_content_type == 'application/x-image':
        try:
            image = Image.open(io.BytesIO(request_body))
            logger.info("Image opened successfully.")
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image).unsqueeze(0)
            logger.info("Image transformed successfully.")
            return image_tensor
        except Exception as e:
            logger.exception("Error processing the input image.")
            raise e
    else:
        logger.error(f"Unsupported content type: {request_content_type}")
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Make the prediction.
    """
    logger.info("In predict_fn.")
    try:
        with torch.no_grad():
            prediction = model(input_data)
        logger.info("Prediction made successfully.")
        return prediction
    except Exception as e:
        logger.exception("Error during prediction.")
        raise e

def output_fn(prediction, content_type):
    """
    Postprocess and return the prediction.
    """
    logger.info("In output_fn.")
    if content_type == 'application/json':
        try:
            response = prediction.cpu().numpy().tolist()  # Ensure prediction is on CPU
            logger.info("Output generated successfully.")
            return response
        except Exception as e:
            logger.exception("Error postprocessing the prediction.")
            raise e
    else:
        logger.error(f"Unsupported content type: {content_type}")
        raise ValueError(f"Unsupported content type: {content_type}")

# This is the handler function
def handler(data, context):
    """
    This function is called when the endpoint is invoked.
    """
    logger.info("Handler invoked.")
    if context.request_content_type == 'application/x-image':
        try:
            data = input_fn(data, context.request_content_type)
            prediction = predict_fn(data, context.model)
            response = output_fn(prediction, context.accept_header)
            logger.info("Handler completed successfully.")
            return response
        except Exception as e:
            logger.exception("Error handling the request.")
            raise e
    else:
        logger.error(f"Unsupported content type: {context.request_content_type}")
        raise ValueError(f"Unsupported content type: {context.request_content_type}")