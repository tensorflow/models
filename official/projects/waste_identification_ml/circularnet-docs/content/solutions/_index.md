CircularNet's open-source nature allows for flexible deployment across various
infrastructures, enabling you to tailor the system to your waste management
needs and resources.

Deployment solutions depend on several factors, including the following:

-  **Technical expertise**: The technical skills available within your team to
   manage the deployment and ongoing maintenance.
-  **Existing infrastructure**: The hardware and software components you already
   have in place.
-  **Data availability needs**: How quickly you need access to the analysis
   results. Options include real-time and batch analysis.
-  **Budget and resources**: The financial and computational resources you can
   allocate to CircularNet.

This page guide you through different deployment options, helping you choose the
one that best aligns with your requirements and capabilities. <br/><br/>

{{< table_of_contents >}}

---

## Where to host the models

While CircularNet's models are freely downloadable from [GitHub](https://github.com/tensorflow/models/tree/master/official/projects/waste_identification_ml), they require a computational environment to perform their analysis. You have the following three main options for where to host and run these models:

-  **Cloud account**: A cloud provider manages the infrastructure for you so
   that you can benefit from the advantages of the cloud, such as paying only
   for how much you consume instead of maintaining your own data center. For
   example, you create a self-managed server hosted on the cloud infrastructure
   to deploy the ML models. The provider handles software updates, the
   underlying network, and data encryption. This guide contains a sample
   solution in [Google Cloud](https://cloud.google.com/gcp), but you can choose
   any other cloud provider to host ML models.
-  **Edge device**: A self-contained physical device equipped with all the
   necessary tools to run the models and perform predictions locally. It offers
   a turnkey solution, handling the entire analysis process on-site. While the
   device can connect to the cloud for dashboard visualization, it doesn't
   require constant internet connectivity.
-  **Your server**: You have complete control over the hardware and software
   environment, allowing full customization. This option is suitable if you have
   the technical expertise and resources to manage your server infrastructure.

The best choice depends on your priorities. An edge device is ideal for
real-time analysis and situations where internet connectivity is limited. A
cloud solution offers scalability and flexibility, while running the models on
your server provides maximum control but requires more technical management.

The following sections describe the advantages and considerations of each
option:

-  [**Cloud deployment**](#cloud-deployment)
-  [**Edge device deployment**](#edge-device-deployment)
-  [**In-house server deployment**](#in-house-server-deployment)

### Cloud deployment

If you prefer a fully managed solution and value flexibility, deploying CircularNet on a cloud platform like [Google Cloud](https://cloud.google.com/) could be ideal. You benefit from the cloud's infrastructure while retaining control over resource configuration and model customization.

This approach is well-suited for those who want the following functionalities:

-  **A hosted solution:** The entire system, including the [dashboard for reporting and visualization](/official/projects/waste_identification_ml/circularnet-docs/content/view-data/), resides in the cloud, eliminating the need for on-site hardware management.
-  **Flexibility and customization:** You can modify models, install additional software or libraries, and experiment with different frameworks as your needs evolve.

Using a cloud account requires technical expertise to manage infrastructure,
commands, and connection settings.

The following are the benefits of cloud deployment:

-  **Customization and experimentation:** Install additional software or
   libraries and try different frameworks to tailor the models to your needs.
-  **Offline data access**: Store and access your captured images privately and
   securely in [Google Cloud Storage](https://cloud.google.com/storage).
-  **Scalability**: A cloud infrastructure can scale compute resources with
   graphics processing units (GPUs) up or down based on your workload. If you
   need more power for complex tasks, you can scale up. On the other hand, if
   you have lower traffic, you can scale down to save costs.
-  **Lower upfront costs**: Avoid the initial investment in hardware and
   deployment setup required for edge devices or on-premise servers. For pricing
   information on Google Cloud, see [Google Cloud pricing](https://cloud.google.com/pricing).
-  **Managed infrastructure**: Google Cloud handles patching, security updates,
   and infrastructure management, freeing you to focus on your model.
-  **High availability**: To minimize downtime for CircularNet's image
   recognition service, you can configure your infrastructure for redundancy and
   automatic failover.

Overall, Google Cloud offers a scalable, flexible, and cost-effective solution
for running the ML model with offline data. It balances processing power,
manageability, and cost efficiency.

To learn how to install CircularNet on Google Cloud, see [Deploy CircularNet](/official/projects/waste_identification_ml/circularnet-docs/content/deploy-cn/).

### Edge device deployment

An edge device brings computing power directly to your facility for on-site data
processing. It requires local installation and setup, providing a comprehensive
solution for hosting and running CircularNet's ML models with a self-managed
experience.

**In most cases, we recommend running the ML model on an [NVIDIA](https://www.nvidia.com/en-us/edge-computing/) edge device.** For information about the minimum hardware requirements for the device to run CircularNet, [choose edge device hardware](/official/projects/waste_identification_ml/circularnet-docs/content/system-req/choose-edge-device/).

The configuration of an edge device has benefits such as the following:

-  **Real-time inference**: Edge devices can perform real-time inference,
   allowing you to process data as it is generated without significant delay.
   This attribute enables the image identification feature to respond quickly to
   user inputs or changes in the environment.
-  **Limited connectivity:** Edge devices can operate in environments with
   limited, unreliable, or non-existent internet connection.
-  **High-performance inference**: Edge devices deliver rapid data processing
   and analysis for low-latency applications.
-  **Energy efficiency**: Edge devices are designed to be power-efficient,
   allowing them to perform complex computations while consuming minimal power.
   This fact makes them ideal for use in battery-powered or energy-constrained
   devices, such as some [machine vision cameras](/official/projects/waste_identification_ml/circularnet-docs/content/system-req/choose-camera/)
   that you can use to capture images.
-  **Compact form**: Edge devices are small and easily integrated into various
   setups, eliminating the need for bulky external hardware.
-  **Compatibility**: Edge devices support multiple frameworks and models. This
   compatibility ensures you can leverage existing models and tools without
   extensive modifications or retraining.

Because of its on-site processing capabilities, an edge device might be
preferable for applications that demand split-second analysis. Due to reduced
network latency, processing on an edge device is fast for real-time
applications.

While edge devices offer numerous advantages, they have less processing power
than larger cloud-based systems and may not be suitable for high-volume or
complex tasks. Additionally, a device failure could temporarily disrupt
operations. Choosing and setting up an edge device requires technical expertise
to manage the hardware and software.

To learn how to install CircularNet on an edge device, see [Deploy CircularNet](/official/projects/waste_identification_ml/circularnet-docs/content/deploy-cn/).

### In-house server deployment

If you have a server, you manage every configuration setting to host the models.
You have complete control over the infrastructure, model customization, and data
management. This alternative is ideal if you count on developers and staff with
a high level of technical expertise and you already have a data center intended
for analysis and waste identification.

Using an in-house server to host the model provides benefits such as the
following:

-  **Full control and customization**: Install libraries, frameworks, or custom
   tools to modify the ML model, optimize the server, and gain complete control
   over the hardware and software environment.
-  **Data privacy**: Keep the model and offline data entirely on-premise.
-  **Cost-effective in the long run (potentially)**: After the initial
   investment in hardware, pay no recurring monthly fees. Depending on use case
   demands, a server can be cost-effective, especially for long-term use with
   consistent workloads.

However, using your server also comes with some drawbacks:

-  **Scalability challenges**: Scaling up resources requires buying additional
   hardware, which can be expensive and time-consuming. Scaling down is also not
   as flexible.
-  **Management overhead**: The user is responsible for maintaining the server
   hardware, updating software, applying security patches, and troubleshooting
   hardware issues.
-  **Limited redundancy**: A single server failure can lead to downtime for the
   image recognition service.

This guide doesn't cover the detailed steps for deploying CircularNet on your
server. The deployment process is highly dependent on your specific on-premise
infrastructure, data center setup, customized settings, available resources, and
the unique requirements of your use case.

You must consult resources and documentation tailored to your specific server environment and adapt [the CircularNet models](https://github.com/tensorflow/models/tree/master/official/projects/waste_identification_ml) to deploy and integrate them successfully into your existing systems. However, if you choose this alternative, we recommend providing a computational unit to run AI models, perform inference, and connect to a [dashboard for data analysis and reporting](/official/projects/waste_identification_ml/circularnet-docs/content/view-data/).