using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Processing;
using SixLabors.Primitives;

namespace OnnxAppService
{
    public class Startup
    {
        public void ConfigureServices(IServiceCollection services)
        {
        }

        public void Configure(IApplicationBuilder app, IHostingEnvironment env)
        {
            app.UseDeveloperExceptionPage();

            var onnxPath = Path.Combine(env.ContentRootPath, "products.onnx");
            var session = new InferenceSession(onnxPath);
            
            app.Run(context =>
            {
                var inputImagePath = Path.Combine(env.ContentRootPath, "drill.jpg");
                var data = ConvertImageToTensor(inputImagePath);
                var input = NamedOnnxValue.CreateFromTensor<float>("data", data);
                using (var output = session.Run(new[] { input }))
                {
                    var prediction = output
                        .First(i => i.Name == "classLabel")
                        .AsEnumerable<string>()
                        .First();
                    return context.Response.WriteAsync(prediction);
                }
            });
        }

        private DenseTensor<float> ConvertImageToTensor(string imagePath)
        {
            var data = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            using (var image = Image.Load(imagePath))
            {
                image.Mutate(ctx => ctx.Resize(new ResizeOptions
                {
                    Size = new Size(224, 224),
                    Mode = ResizeMode.Stretch
                }));
                for (var x = 0; x < image.Width; x++)
                {
                    for (var y = 0; y < image.Height; y++)
                    {
                        var color = image.GetPixelRowSpan(y)[x];
                        data[0, 0, x, y] = color.B;
                        data[0, 1, x, y] = color.G;
                        data[0, 2, x, y] = color.R;
                    }
                }
            }
            return data;
        }
    }
}
