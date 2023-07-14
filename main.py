from get_data import *
from network import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pandas as pd
import numpy as np
directory = './top_300_metrics/'

projects=["home-assistant/core","NixOS/nixpkgs","microsoft/vscode","flutter/flutter","MicrosoftDocs/azure-docs","dotnet/runtime","pytorch/pytorch","odoo/odoo","element-fi/elf-council-frontend","godotengine/godot","rust-lang/rust","elastic/kibana","archway-network/testnets","kubernetes/kubernetes","grafana/grafana","microsoft/winget-pkgs","microsoft/PowerToys","solana-labs/token-list","firstcontributions/first-contributions","taozhiyu/TyProAction","Homebrew/homebrew-core","zephyrproject-rtos/zephyr","dotnet/roslyn","WordPress/gutenberg","python/cpython","elementor/elementor","ClickHouse/ClickHouse","mdn/content","cypress-io/cypress","AdguardTeam/AdguardFilters","vercel/next.js","expo/expo","kubernetes/website","PaddlePaddle/Paddle","ray-project/ray","apache/airflow","mui/material-ui","aws/aws-cdk","ValveSoftware/Dota2-Gameplay","huggingface/transformers","google/it-cert-automation-practice","openshift/openshift-docs","DefinitelyTyped/DefinitelyTyped","dotnet/aspnetcore","tensorflow/tensorflow","microsoft/vcpkg","golang/go","nodejs/node","elastic/elasticsearch","flutter/engine","microsoft/playwright","Automattic/wp-calypso","airbytehq/airbyte","dotnet/maui","hashicorp/terraform-provider-azurerm","Homebrew/homebrew-cask","tgstation/tgstation","webcompat/web-bugs","microsoft/TypeScript","Azure/azure-sdk-for-net","microsoftgraph/microsoft-graph-docs","hashicorp/terraform-provider-aws","ant-design/ant-design","CleverRaven/Cataclysm-DDA","qmk/qmk_firmware","pandas-dev/pandas","yt-dlp/yt-dlp","ShadowMario/FNF-PsychEngine","Azure/azure-cli","angular/angular","quarkusio/quarkus","facebook/react-native","PrestaShop/PrestaShop","Koenkk/zigbee2mqtt","istio/istio","apache/superset","brave/brave-core","nrwl/nx","education/GitHubGraduation-2022","conan-io/conan-center-index","envoyproxy/envoy","LeetCode-Feedback/LeetCode-Feedback","nextcloud/server","wjz304/Redpill_CustomBuild","element-plus/element-plus","ultralytics/yolov5","getsentry/sentry","Lightning-AI/lightning","qgis/QGIS","idsb3t1/KEEP-pipeline-tests-resources","microsoft/onnxruntime","Chia-Network/chia-blockchain","prisma/prisma","Azure/azure-rest-api-specs","cockroachdb/cockroach","strapi/strapi","gitpod-io/gitpod","llvm/llvm-project","trinodb/trino","systemd/systemd","openjournals/joss-reviews","openvinotoolkit/openvino","go-gitea/gitea","magento/magento2","void-linux/void-packages","openjdk/jdk","TP-Lab/tokens","microsoft/fluentui","uBlockOrigin/uAssets","gradle/gradle","ceph/ceph","symfony/symfony","qbittorrent/qBittorrent","trustwallet/assets","apache/spark","backstage/backstage","gentoo/gentoo","microsoft/terminal","mui/mui-x","Azure/azure-sdk-for-java","woocommerce/woocommerce","sourcegraph/sourcegraph","Ultimaker/Cura","neovim/neovim","bitcoin/bitcoin","metabase/metabase","tachiyomiorg/tachiyomi-extensions","joomla/joomla-cms","dbeaver/dbeaver","mdn/translated-content","files-community/Files","home-assistant/home-assistant.io","mozilla-mobile/fenix","Expensify/App","openwrt/openwrt","MicrosoftDocs/msteams-docs","spack/spack","nuxt/framework","github/docs","MetaMask/metamask-extension","storybookjs/storybook","spyder-ide/spyder","electron/electron","helium/denylist","RocketChat/Rocket.Chat","MicrosoftDocs/microsoft-365-docs","project-chip/connectedhomeip","vitejs/vite","laravel/framework","scikit-learn/scikit-learn","keycloak/keycloak","department-of-veterans-affairs/va.gov-team","apache/shardingsphere","containers/podman","solana-labs/solana","apache/arrow","conda-forge/staged-recipes","jlord/patchwork","ccxt/ccxt","sveltejs/kit","apache/pulsar","logseq/logseq","apple/swift","openssl/openssl","renovatebot/renovate","hashicorp/vault","facebook/react","bitnami/charts","brave/brave-browser","section-engineering-education/engineering-education","microsoft/vscode-jupyter","Azure/azure-sdk-for-python","cms-sw/cmssw","appsmithorg/appsmith","AUTOMATIC1111/stable-diffusion-webui","dotnet/docs","JetBrains/swot","openshift/release","microsoft/WSL","ValveSoftware/Proton","apache/hudi","rancher/rancher","openhab/openhab-addons","SerenityOS/serenity","home-assistant/frontend","pingcap/tidb","metersphere/metersphere","zulip/zulip","JuliaLang/julia","cilium/cilium","dotnet/efcore","JuliaRegistries/General","PaddlePaddle/PaddleOCR","argoproj/argo-cd","RPCS3/rpcs3","espressif/esp-idf","apache/flink","raycast/extensions","tailscale/tailscale","Azure/azure-powershell","grpc/grpc","nrfconnect/sdk-nrf","microsoft/azuredatastudio","MetaMask/eth-phishing-detect","scipy/scipy","directus/directus","demisto/content","ArduPilot/ardupilot","MarlinFirmware/Marlin","rails/rails","ppy/osu","aws-amplify/amplify-cli","JacksonKearl/testissues","desktop/desktop","ankidroid/Anki-Android","Azure/azure-sdk-for-js","bioconda/bioconda-recipes","bevyengine/bevy","apache/beam","open-telemetry/opentelemetry-collector-contrib","leanprover-community/mathlib","remix-run/remix","github/codeql","OpenAPITools/openapi-generator","obsproject/obs-studio","DataDog/datadog-agent","ethereum/ethereum-org-website","cloudflare/cloudflare-docs","grafana/loki","apache/iceberg","gatsbyjs/gatsby","gravitational/teleport","darktable-org/darktable","apache/tvm","open-mmlab/mmdetection","azerothcore/azerothcore-wotlk","TeamNewPipe/NewPipe","denoland/deno","apache/dolphinscheduler","matplotlib/matplotlib","type-challenges/type-challenges","postmanlabs/postman-app-support","google-test/signclav2-probe-repo","matrix-org/synapse","firebase/flutterfire","xamarin/xamarin-macios","opencv/opencv","flathub/flathub","vectordotdev/vector","taosdata/TDengine","ruffle-rs/ruffle","termux/termux-packages","Automattic/jetpack","dotnet/AspNetCore.Docs","freddier/hyperblog","oppia/oppia","Skyrat-SS13/Skyrat-tg","macports/macports-ports","Kaiserreich/Kaiserreich-4","apache/doris","flutter/plugins","rapid7/metasploit-framework","xbmc/xbmc","jitsi/jitsi-meet","PixelExperience/android-issues","ziglang/zig","firebase/firebase-android-sdk","mastodon/mastodon","PowerShell/PowerShell","docker/docs","coolsnowwolf/lede","prusa3d/PrusaSlicer","redis/redis","zero-to-mastery/start-here-guidelines","ansible/ansible","tachiyomiorg/tachiyomi","alibaba/nacos","newrelic/docs-website","kubernetes/minikube","yuzu-emu/yuzu","o3de/o3de","kubernetes/test-infra","GoogleChrome/developer.chrome.com","lensapp/lens","filecoin-project/filecoin-plus-large-datasets","Regalis11/Barotrauma","rstudio/rstudio","mlflow/mlflow","angular/components","kubevirt/kubevirt","nextcloud/desktop","apache/apisix","IntelRealSense/librealsense","mrdoob/three.js","flybywiresim/a32nx","helix-editor/helix","php/php-src","unifyai/ivy","influxdata/telegraf","mattermost/mattermost-webapp"]
# class_1=["activity.json","attention.json","bus_factor.json","change_requests.json","change_requests_reviews.json","code_change_lines_add.json","code_change_lines_remove.json","code_change_lines_sum.json","inactive_contributors.json","issue_comments.json","issues_and_change_request_active.json","issues_closed.json","issues_new.json","new_contributors.json","openrank.json","participants.json","stars.json","technical_fork.json"]
# class_2=["change_request_age.json","change_request_resolution_duration.json","change_request_response_time.json","issue_age.json","issue_resolution_duration.json","issue_response_time.json"]
class_1=["bus_factor.json","change_requests.json","change_requests_reviews.json","code_change_lines_add.json","code_change_lines_remove.json","code_change_lines_sum.json","inactive_contributors.json","issues_and_change_request_active.json","issues_closed.json","issues_new.json","new_contributors.json","openrank.json","technical_fork.json"]
class_2=["change_request_age.json","change_request_resolution_duration.json","change_request_response_time.json","issue_age.json","issue_resolution_duration.json","issue_response_time.json",]

def rf_model(X_train_all,y_train_all,X_test_all,y_test_all,feature_names):
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    # 在训练数据上拟合模型
    rf.fit(X_train_all, y_train_all)
    mse_list = []
    for X_test,y_test in zip(X_test_all,y_test_all):
        rf_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, rf_pred)
        mse_list.append(mse)
    mean_mse = np.mean(mse_list)
    print("Random Forest MSE:", mean_mse)
    importances = rf.feature_importances_ 
    sorted_idx = np.argsort(importances)
    print("Feature ranking:")

    for i in sorted_idx[::-1]:
        name = feature_names[i]
        value = importances[i]
        print(f"{name}: {value:.5f}")
    return mean_mse

def xgb_model(X_train_all,y_train_all,X_test_all,y_test_all):
    model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1, max_depth = 5, n_estimators = 100)
    model.fit(X_train_all, y_train_all)
    mse_list = []
    for X_test,y_test in zip(X_test_all,y_test_all):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)
    mean_mse = np.mean(mse_list)
    print("XGBoost MSE:", mean_mse)
    return mean_mse

def main():
    df_list=get_data(directory,projects,class_1,class_2)
    X_train_all,y_train_all,X_test_all,y_test_all,feature_names,all_features=data_process(df_list)
    # rf_mse=rf_model(X_train_all,y_train_all,X_test_all,y_test_all,feature_names)
    # xgb_mse=xgb_model(X_train_all,y_train_all,X_test_all,y_test_all)
    train_loaders,val_loaders,test_loaders=get_data_loader(df_list,all_features)
    model = LSTM(input_size=74, hidden_size=50, num_layers=2, output_size=1)
    model = model.float().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    train_model(model, train_loaders, val_loaders, criterion, optimizer, num_epochs=100,early_stop=3,PATH = './test_state_dict.pth')
    test_predictions = []
    test_targets = []
    for data, targets in test_loaders:
        data = data.float().to(device)
        targets = targets.float().unsqueeze(1).to(device)
        outputs = model(data)
        test_predictions.extend(outputs.detach().cpu().numpy())
        test_targets.extend(targets.detach().cpu().numpy())
    test_mse = mean_squared_error(test_targets, test_predictions)
    print("LSTM MSE:", test_mse)


if __name__ == "__main__":
    main()