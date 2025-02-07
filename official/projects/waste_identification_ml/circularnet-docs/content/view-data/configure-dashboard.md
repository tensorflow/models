Follow these steps to configure a Looker Studio dashboard and visualize the
model results for data analysis and reporting:

1. Open the public Looker Studio template.
1. Click **More options** > **Make a copy**.
1. On the **Copy this report** window, enter the following details:

    1. Leave the value on the **Original data source** menu as shown because it
       is the value of the template's BigQuery data source.
    1. On the **New data source** menu, select the name of the BigQuery table
       that contains the model results you want to use as the data source for
       the dashboard.

1. If the **New data source** menu doesn't show your BigQuery table as an
   available data source, perform the following actions:

    1. Click **Create data source** on the **New data source** menu.
    1. On the **Google connector** page, select **BigQuery**.
    1. If prompted, click **Authorize**.
    1. Select the Google Cloud project that hosts your BigQuery dataset.
    1. Select the dataset and table you want to use.
    1. Click **Connect**.
    1. On the new page, click **Add to report**.

1. Click **Copy report**.

A copy of the template dashboard is created in your Google Cloud account. You can change the title and customize this copy to fit your needs. For information about customizing and using the Looker Studio dashboard, see the [Quick start guide of Looker Studio](https://support.google.com/looker-studio/answer/9171315) and the [Looker documentation](https://cloud.google.com/looker/docs/intro).