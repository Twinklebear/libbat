'use strict';

var gl = null;
var canvas = null;
var proj = null;
var camera = null;
var projView = null;

var datasets = null;
var selectedDataset = null;

var vao = null;
var pointDataVbo = null;
var attribDataVbo = null;

var firstUpload = true;
var newBATUpload = true;
var pointShader = null;
var positions = null;
var attributes = null;
var attributeRange = [0, 1];
var numLoadedPoints = 0;

var pointRadiusSlider = null;
var qualitySlider = null;
var attributeList = null;

var prevQuality = 0.0;
var invalidPrevQuality = true;

var WIDTH = 640;
var HEIGHT = 480;

const defaultEye = vec3.set(vec3.create(), 0.5, 0.5, 1.5);
const center = vec3.set(vec3.create(), 0.5, 0.5, 0.5);
const up = vec3.set(vec3.create(), 0.0, 1.0, 0.0);

var colormaps = {
    "Cool Warm": "colormaps/cool-warm-paraview.png",
    "Matplotlib Plasma": "colormaps/matplotlib-plasma.png",
    "Matplotlib Virdis": "colormaps/matplotlib-virdis.png",
    "Rainbow": "colormaps/rainbow.png",
    "Samsel Linear Green": "colormaps/samsel-linear-green.png",
    "Samsel Linear YGB 1211G": "colormaps/samsel-linear-ygb-1211g.png",
};

var loadBATDataset = function(dataset, onload) {
    var loadingProgressText = document.getElementById("loadingText");
    var loadingProgressBar = document.getElementById("loadingProgressBar");
    loadingProgressText.innerHTML = "Loading Dataset";
    loadingProgressBar.setAttribute("style", "width: 0%");

    var url = "http://localhost:1234/dataset/" + dataset;
    var req = new XMLHttpRequest();

    req.open("POST", url, true);
    req.responseType = "arraybuffer";
    req.onprogress = function(evt) {
        var percent = evt.loaded / evt.total * 100;
        loadingProgressBar.setAttribute("style", "width: " + percent.toFixed(2) + "%");
    };
    req.onerror = function(evt) {
        console.log("Error!?");
        console.log(req);
        console.log(evt);
        loadingProgressText.innerHTML = "Error Loading Dataset";
        loadingProgressBar.setAttribute("style", "width: 0%");
    };
    req.onload = function(evt) {
        loadingProgressText.innerHTML = "Loaded Dataset";
        loadingProgressBar.setAttribute("style", "width: 100%");
        var buffer = req.response;
        if (buffer) {
            onload(buffer);
        } else {
            alert("Unable to load buffer properly from volume?");
            console.log("no buffer?");
        }
    };

    var range = $("#slider-range").slider("option", "values");
    
    var query_params = {
        quality: parseFloat(qualitySlider.value),
        prev_quality: invalidPrevQuality ? 0 : prevQuality,
        attribute: attributeList.value,
        range_min: parseFloat(range[0]),
        range_max: parseFloat(range[1]),
    };
    req.send(JSON.stringify(query_params));
}

var selectBATDataset = function(shouldClearCache, resetRangeSlider) {
    var selection = document.getElementById("datasets").value;
    var loadingInfo = document.getElementById("loadingInfo");

    loadingInfo.style.display = "block";

    for (var i = 0; i < datasets.length; ++i) {
        if (datasets[i]["name"] == selection) {
            if (selectedDataset != datasets[i]) {
                newBATUpload = true;
            }
            selectedDataset = datasets[i];
        }
    }

    if (newBATUpload) {
        clearCache();
        fillAttributeSelector();
    } else if (resetRangeSlider) {
        clearCache();
        console.log("reset slider");
        for (var i = 0; i < selectedDataset["attributes"].length; ++i) {
            if (selectedDataset["attributes"][i]["name"] == attributeList.value) {
                attributeRange = selectedDataset["attributes"][i]["range"];
            }
        }
        console.log("setting to ");
        console.log(attributeRange);
        var step_size = (parseFloat(attributeRange[1])-parseFloat(attributeRange[0]))/100.0;
        $("#slider-range").slider("option", "values", [attributeRange[0], attributeRange[1]]);
        $("#range_value").html(attributeRange[0] + " to " + attributeRange[1]);
        $("#slider-range").slider("option", "min", parseFloat(attributeRange[0])-step_size)
        $("#slider-range").slider("option", "max", parseFloat(attributeRange[1])+step_size)
        $("#slider-range").slider("option", "step", step_size)
    }

    document.getElementById("datasetInfo").innerHTML = JSON.stringify(selectedDataset);

    loadBATDataset(selection, function(dataBuffer) {

        if (shouldClearCache) {
            clearCache();
        }
        var header = new Uint32Array(dataBuffer, 0, 1);
        var currLoadedPoints = header[0];

        if(positions == null) {
            numLoadedPoints = currLoadedPoints;
          positions = new Buffer(currLoadedPoints * 3, "float32");
        } else {
            numLoadedPoints += currLoadedPoints;
        }

        positions.append(new Float32Array(dataBuffer, 4, currLoadedPoints * 3));

        if(attributes == null)
          attributes = new Buffer(currLoadedPoints, "float32");

        attributes.append(new Float32Array(dataBuffer, 4 + currLoadedPoints * 3 * 4, currLoadedPoints))

        gl.bindVertexArray(vao);

        gl.bindBuffer(gl.ARRAY_BUFFER, pointDataVbo);
        gl.bufferData(gl.ARRAY_BUFFER, positions.buffer, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, attribDataVbo);
        gl.bufferData(gl.ARRAY_BUFFER, attributes.buffer, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);

        document.getElementById("numPoints").innerHTML = numLoadedPoints + " new("+currLoadedPoints+")";

        for (var i = 0; i < selectedDataset["attributes"].length; ++i) {
            if (selectedDataset["attributes"][i]["name"] == attributeList.value) {
                attributeRange = selectedDataset["attributes"][i]["range"];
            }
        }

        // step size dividing the range interval in 100 steps
        var step_size = (parseFloat(attributeRange[1])-parseFloat(attributeRange[0]))/100.0;

        // TODO fix this bug, when we set the extact min and max of the range the slider
        // does not always reach those values (and so the query never reached the min and max)
        // this ugly workaround now expands the range in the slider 
        if (newBATUpload) {
            $("#slider-range").slider("option", "min", parseFloat(attributeRange[0])-step_size)
            $("#slider-range").slider("option", "max", parseFloat(attributeRange[1])+step_size)
            $("#slider-range").slider("option", "step", step_size)
            console.log("reset sliders for newbat");
            $("#slider-range").slider("option", "values", [attributeRange[0] - step_size, attributeRange[1] + step_size]);
            $("#range_value").html(attributeRange[0] + " to " + attributeRange[1]);
        }

        if (firstUpload) {
            firstUpload = false;
            setInterval(function() {
                // Save them some battery if they're not viewing the tab
                if (document.hidden) {
                    return;
                }

                // Reset the sampling rate and camera for new volumes
                if (newBATUpload) {
                    camera = new ArcballCamera(defaultEye, center, up, 100, [WIDTH, HEIGHT]);
                    camera.zoom(-30);
                }
                projView = mat4.mul(projView, proj, camera.camera);

                gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

                pointShader.use(gl);
                gl.uniform1i(pointShader.uniforms["colormap"], 0);
                gl.uniformMatrix4fv(pointShader.uniforms["proj_view"], false, projView);
                gl.uniform3fv(pointShader.uniforms["eye_pos"], camera.eyePos());
                gl.uniform1f(pointShader.uniforms["radius_scale"], pointRadiusSlider.value);
                gl.uniform2f(pointShader.uniforms["value_range"], attributeRange[0],
                    attributeRange[1]);
                gl.drawArrays(gl.POINTS, 0, numLoadedPoints);

                // Wait for rendering to actually finish so we can time it
                gl.finish();
                newBATUpload = false;
            }, 32);
        }
    });
}

var selectColormap = function() {
    var selection = document.getElementById("colormapList").value;
    var colormapImage = new Image();
    colormapImage.onload = function() {
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 180, 1,
            gl.RGBA, gl.UNSIGNED_BYTE, colormapImage);
    };
    colormapImage.src = colormaps[selection];
}

function clearCache(){
    console.log("clear cache");
    positions = null;
    attributes = null;
}

window.onload = function() {
    fillColormapSelector();

    pointRadiusSlider = document.getElementById("pointRadiusSlider");
    pointRadiusSlider.value = 5;

    qualitySlider = document.getElementById("qualitySlider");
    qualitySlider.value = 0.05;

    attributeList = document.getElementById("attributeList");

    // range slider initialization
    $( "#slider-range" ).slider({
      range: true,
      min: 0.0,
      max: 1.0,
      step: 0.1,
      values: [ 0.0, 1.0 ],
      slide: function( event, ui ) { // handle the updates of the range slider
        $( "#amount" ).val( "$" + ui.values[ 0 ] + " - $" + ui.values[ 1 ] );
        // update range interval text 
        $("#range_value").html(parseFloat(ui.values[ 0 ]) +" - "+parseFloat(ui.values[ 1 ]));
        selectBATDataset(true, false);
      },
    });
    $("#range_value").html("0 - 1");

    canvas = document.getElementById("glcanvas");
    gl = canvas.getContext("webgl2");
    if (!gl) {
        alert("Unable to initialize WebGL2. Your browser may not support it");
        return;
    }

    var extensions = ["EXT_color_buffer_float", "EXT_float_blend"];
    for (var i = 0; i < extensions.length; ++i) {
        if (!getGLExtension(gl, extensions[i])) {
            alert("Required WebGL extensions missing (" + extensions[i] + "), aborting");
            return;
        }
    }

    WIDTH = canvas.width;
    HEIGHT = canvas.height;

    proj = mat4.perspective(mat4.create(), 60 * Math.PI / 180.0,
        WIDTH / HEIGHT, 0.1, 500);
    projView = mat4.create();

    camera = new ArcballCamera(defaultEye, center, up, 2, [WIDTH, HEIGHT]);

    // Register mouse and touch listeners
    var controller = new Controller();

    controller.mousemove = function(prev, cur, evt) {
        if (evt.buttons == 1) {
            camera.rotate(prev, cur);
        } else if (evt.buttons == 2) {
            camera.pan([cur[0] - prev[0], prev[1] - cur[1]]);
        }
    };
    controller.wheel = function(amt) { camera.zoom(amt / 4.0); };
    controller.pinch = controller.wheel;
    controller.twoFingerDrag = function(drag) { camera.pan(drag); };

    controller.registerForCanvas(canvas);

    gl.enable(gl.DEPTH_TEST);

    gl.clearDepth(1.0);
    gl.clearColor(1.0, 1.0, 1.0, 1.0);

    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    pointDataVbo = gl.createBuffer();
    attribDataVbo = gl.createBuffer();

    // TODO: Similar to the desktop viewer we need one shader per attrib-type
    pointShader = new Shader(gl, vertShader, fragShader);

    // Load the default colormap, after which we populate the data set selector
    // to let users pick a dataset to view
    var colormapImage = new Image();
    colormapImage.onload = function() {
        var colormap = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, colormap);
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, 180, 1);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 180, 1,
            gl.RGBA, gl.UNSIGNED_BYTE, colormapImage);

        fillDatasetSelector();
    };
    colormapImage.src = "colormaps/cool-warm-paraview.png";
}

var fillDatasetSelector = function() {
    // Make a request to the server to get the list of BAT files available
    var url = "http://localhost:1234/datasets";
    var req = new XMLHttpRequest();

    req.open("GET", url, true);
    req.responseType = "application/json";
    req.onerror = function() {
        alert("Failed to get dataset list from the server");
    };
    req.onload = function(evt) {
        datasets = JSON.parse(req.response);

        var selector = document.getElementById("datasets");
        for (var i = 0; i < datasets.length; ++i) {
            var bat = datasets[i];
            var opt = document.createElement("option");
            opt.value = bat["name"];
            opt.innerHTML = bat["name"];
            selector.appendChild(opt);
        }

        // TODO: Later this should make a different kind of request, to just update
        // the quality level

        // Get the value when input gains focus
        $("#qualitySlider").focus(function(){
            prevQuality = parseFloat($("#qualitySlider").val());
            console.log("previous", prevQuality)
        });
        
        $("#qualitySlider").on("input change", function(event) { 
            if(parseFloat($("#qualitySlider").val()) <= prevQuality){
                invalidPrevQuality=true;
                clearCache()
            }
            else
                invalidPrevQuality=false;

            selectBATDataset(); 
        });
        selectBATDataset();
    };
    req.send();
}

var fillAttributeSelector = function() {
    var selector = document.getElementById("attributeList");
    selector.innerHTML = "";
    for (var i = 0; i < selectedDataset["attributes"].length; ++i) {
        var opt = document.createElement("option");
        opt.value = selectedDataset["attributes"][i]["name"];
        opt.innerHTML = opt.value;
        selector.appendChild(opt);
    }
}

var fillColormapSelector = function() {
    var selector = document.getElementById("colormapList");
    for (var c in colormaps) {
        var opt = document.createElement("option");
        opt.value = c;
        opt.innerHTML = c;
        selector.appendChild(opt);
    }
}

