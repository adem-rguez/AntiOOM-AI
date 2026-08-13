use proto::daemon_service_client::DaemonServiceClient;
use proto::{GenerateRequest, GenerateResponse, LoadModelRequest, LoadModelResponse, StatusRequest, StatusResponse};
use tonic::transport::Channel;

pub struct DaemonClient {
    inner: DaemonServiceClient<Channel>,
}

impl DaemonClient {
    pub async fn connect<D>(dst: D) -> Result<Self, tonic::transport::Error>
    where
        D: TryInto<tonic::transport::Endpoint>,
        D::Error: Into<tonic::codegen::StdError>,
    {
        let client = DaemonServiceClient::connect(dst).await?;
        Ok(Self { inner: client })
    }

    pub async fn generate(
        &mut self,
        request: GenerateRequest,
    ) -> Result<GenerateResponse, tonic::Status> {
        let response = self.inner.generate(request).await?;
        Ok(response.into_inner())
    }

    pub async fn load_model(
        &mut self,
        request: LoadModelRequest,
    ) -> Result<LoadModelResponse, tonic::Status> {
        let response = self.inner.load_model(request).await?;
        Ok(response.into_inner())
    }

    pub async fn get_status(&mut self) -> Result<StatusResponse, tonic::Status> {
        let response = self.inner.get_status(StatusRequest {}).await?;
        Ok(response.into_inner())
    }
}
