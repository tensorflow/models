Before deploying CircularNet, you must follow these steps. Depending on whether
you want to deploy CircularNet on Google Cloud or an NVIDIA edge device, choose
one of the following options: <br><br>

<details>
  <summary>
    Google Cloud
  </summary>
  <ol>
    <li><a href="https://console.cloud.google.com/">Create a Google Cloud account</a>.</li>
    <li><a href="https://cloud.google.com/cloud-console">Open the Google Cloud console</a>.</li>
    <li><a href="https://cloud.google.com/resource-manager/docs/creating-managing-projects">Create a project on your Google Cloud account</a>.</li>
    <li><a href="https://cloud.google.com/billing/docs/how-to/create-billing-account">Set up your Cloud Billing account</a> to manage your Google Cloud spending.</li>
    <li><p>Enable the following APIs:</p><br>
        <ul>
            <li>Compute Engine API</li>
            <li>BigQuery API</li>
            <li>Cloud Storage API</li>
        </ul>
        <p>To enable APIs, see <a href="https://cloud.google.com/endpoints/docs/openapi/enable-api">Enabling an API in your Google Cloud project</a>.</p><br>
    </li>
    <li><p><a href="https://cloud.google.com/compute/docs/gpus/create-gpu-vm-general-purpose">Create a Compute Engine virtual machine (VM) that has attached an NVIDIA T4 GPU</a>. Use the following settings on your VM:</p><br>
        <ul>
            <li><strong>Machine configuration</strong>: GPUs</li>
            <li><strong>GPU type</strong>: NVIDIA T4</li>
            <li><strong>Number of GPUs</strong>: 1</li>
            <li><strong>Machine type</strong>: n1-standard-8 (8 vCPU, 4 core, 30 GB memory)</li>
            <li><strong>Boot disk</strong>: A size of 300 GB</li>
            <li><strong>Identity and API access</strong>: Allow full access to all Cloud APIs</li>
            <li><strong>Firewall</strong>: Allow HTTP and HTTPS traffic</li>
        </ul>
        <p><strong>Note</strong>: Give your VM a name that is easy to remember and deploy in a region and a zone close to your physical location that allows GPUs.</p><br>
    </li>
    <li>From the <strong>Navigation menu</strong> on the Google Cloud console, select <strong>Compute Engine</strong> > <strong>VM instances</strong>.</li>
    <li>On the <strong>VM instances</strong> page, find the VM instance you created with the NVIDIA T4 GPU.</li>
    <li>Click <strong>SSH</strong> in the row of the instance that you want to connect to. Let the <strong>SSH-in-browser</strong> tool open. For more information, see <a href="https://cloud.google.com/compute/docs/connect/standard-ssh#connect_to_vms">Connect to VMs</a>.
    <p><strong>Important</strong>: If the SSH connection fails, you must create a firewall rule for the VM to allow TCP ingress traffic. <a href="https://cloud.google.com/iap/docs/using-tcp-forwarding#create-firewall-rule">Use Identity-Aware Proxy (IAP) for TCP forwarding</a> to allow ingress traffic from the IPv4 ranges <code>35.235.240.0/20</code>, <code>0.0.0.0/0</code>, and <code>0.0.0.0/22</code> on TCP ports <code>22,3389</code>.</p><br>
    </li>
  </ol>
</details>

<details>
  <summary>
    Edge device
  </summary>
  <ol>
    <li>Get an edge device, configure it, and connect it to your local machine.</li>
    <li>Ensure you have an internet connection for the deployment and package installation.</li>
    <li>Set up your device, install the software and dependencies you want, and prepare the corresponding developer kit. For information on essential configurations, refer to the software documentation of your edge device. For example, see <a href="https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/index.html">the Developer Guide of NVIDIA Jetson devices</a>.</li>
    <li>Open the terminal of your edge device to interact with the operating system through the command line.</li>
  </ol>
</details>
