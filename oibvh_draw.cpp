#include "oibvh_draw.h"

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "glad/glad.h"
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "utils.h"

// set to 0 if you want to dump mesh file to file, otherwise set to one if you just want to show bvhs
#define RENDER_AABBS_WITH_LINES 1

// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific params methods
enum camera_movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

// default camera values
const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 2.5f;
const float SENSITIVITY = 0.1f;
const float ZOOM = 45.0f;

class Camera {
public:
    // Camera Attributes
    glm::vec3 position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    // Euler Angles
    float Yaw;
    float Pitch;
    // Camera options
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;

    // Constructor with vectors
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH)
        : Front(glm::vec3(0.0f, 0.0f, -1.0f))
        , MovementSpeed(SPEED)
        , MouseSensitivity(SENSITIVITY)
        , Zoom(ZOOM)
    {
        position = position;
        WorldUp = up;
        Yaw = yaw;
        Pitch = pitch;
        updateCameraVectors();
    }
    // Constructor with scalar values
    Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch)
        : Front(glm::vec3(0.0f, 0.0f, -1.0f))
        , MovementSpeed(SPEED)
        , MouseSensitivity(SENSITIVITY)
        , Zoom(ZOOM)
    {
        position = glm::vec3(posX, posY, posZ);
        WorldUp = glm::vec3(upX, upY, upZ);
        Yaw = yaw;
        Pitch = pitch;
        updateCameraVectors();
    }

    // Returns the view matrix calculated using Euler Angles and the LookAt Matrix
    glm::mat4 GetViewMatrix()
    {
        return glm::lookAt(position, position + Front, Up);
    }

    // Processes params received from any keyboard-like params system. Accepts params parameter in the form of camera defined ENUM (to abstract it from windowing systems)
    void ProcessKeyboard(camera_movement direction, float delta_time)
    {
        float velocity = MovementSpeed * delta_time;
        if (direction == FORWARD)
            position += Front * velocity;
        if (direction == BACKWARD)
            position -= Front * velocity;
        if (direction == LEFT)
            position -= Right * velocity;
        if (direction == RIGHT)
            position += Right * velocity;
    }

    // Processes params received from a mouse params system. Expects the offset value in both the x and y direction.
    void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true)
    {
        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

        Yaw += xoffset;
        Pitch += yoffset;

        // Make sure that when pitch is out of bounds, screen doesn't get flipped
        if (constrainPitch) {
            if (Pitch > 89.0f)
                Pitch = 89.0f;
            if (Pitch < -89.0f)
                Pitch = -89.0f;
        }

        // Update Front, Right and Up Vectors using the updated Euler angles
        updateCameraVectors();
    }

    // Processes params received from a mouse scroll-wheel event. Only requires params on the vertical wheel-axis
    void ProcessMouseScroll(float yoffset)
    {
        if (Zoom >= 1.0f && Zoom <= 45.0f)
            Zoom -= yoffset;
        if (Zoom <= 1.0f)
            Zoom = 1.0f;
        if (Zoom >= 45.0f)
            Zoom = 45.0f;
    }

private:
    // Calculates the front vector from the Camera's (updated) Euler Angles
    void updateCameraVectors()
    {
        // Calculate the new Front vector
        glm::vec3 front;
        front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        front.y = sin(glm::radians(Pitch));
        front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        Front = glm::normalize(front);
        // Also re-calculate the Right and Up vector
        Right = glm::normalize(glm::cross(Front, WorldUp)); // Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        Up = glm::normalize(glm::cross(Right, Front));
    }
};

struct bvh_data_draw_call_info_t {
    std::vector<GLuint> m_indexBufSegmentCapacity;
    std::vector<GLsizei> m_indexBufBaseOffset;
    // start and end parameters for glDrawRangeElements
    std::vector<std::pair<GLuint, GLuint> > m_indexBufVertexRange;

    std::vector<uint32_t> leaf_counts;
};

std::vector<glm::vec3> colors = {
    { 0, 1, 0 }, { 0, 0, 1 }, { 1, 1, 0 }, { 0, 1, 1 }, { 1, 0, 1 }
};
int window_width = 1920;
int window_height = 1080;
float lastX = window_width / 2.0f;
float lastY = window_height / 2.0f;
bool firstMouse = true;
// timing
float delta_time = 0.0f; // time between current frame and last frame
float prev_time = 0.0f;
// how much bvh nodes to see
float bvhNodeVisualisationRatio = 1.0;
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
GLFWwindow* window = nullptr;
bool wireframe = false;

GLuint create_shader(GLenum type, const char* src)
{
    GLuint shader;
    GLint shader_ok;
    GLsizei log_length;
    char info_log[8192];

    shader = glCreateShader(type);

    if (!shader) {
        fprintf(stderr, "ERROR: failed to create shader object\n");
        exit(1);
    }

    glShaderSource(shader, 1, (const GLchar**)&src, NULL);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &shader_ok);

    if (!shader_ok) {
        fprintf(stderr, "ERROR: Failed to compile %s shader\n",
            (type == GL_FRAGMENT_SHADER) ? "fragment" : "vertex");

        glGetShaderInfoLog(shader, 8192, &log_length, info_log);

        fprintf(stderr, "BUILD LOG: \n%s\n\n", info_log);

        glDeleteShader(shader);

        std::exit(1);
    }
    return shader;
}

GLuint create_shader_program(int32_t shader_count, ...)
{
    GLint program_ok;
    GLint program = glCreateProgram();

    if (!program) {
        fprintf(stderr, "EEROR: failed to create shader program object\n");
        exit(1);
    }

    va_list arg_list;
    va_start(arg_list, shader_count);
    for (int32_t shader_iter = 0; shader_iter < shader_count; ++shader_iter) {
        glAttachShader(program, va_arg(arg_list, GLint));
    }
    va_end(arg_list);

    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &program_ok);

    if (!program_ok) {
        printf("failed to link shader program\n");

        GLsizei log_length;
        char info_log[8192];

        glGetProgramInfoLog(program, 8192, &log_length, info_log);

        printf("\n%s\n\n", info_log);
        glDeleteProgram(program);

        std::exit(1);
    }
    return program;
}

int graphics_create_program(const std::string& vs_name,
    const std::string& fs_name)
{
    std::string vsrc = read_text_file(vs_name);
    GLint vs = create_shader(GL_VERTEX_SHADER, vsrc.c_str());
    std::string fsrc = read_text_file(fs_name);
    GLint fs = create_shader(GL_FRAGMENT_SHADER, fsrc.c_str());

    GLint program = create_shader_program(2, vs, fs);

    glDeleteShader(vs);

    glDeleteShader(fs);

    return program;
}

void make_cube(std::vector<glm::vec3>& vertices,
    std::vector<unsigned int>& indices, float size_x, float size_y,
    float size_z)
{

    // front quad
    vertices.push_back(glm::vec3(-size_x, -size_y, size_z)); // 0
    vertices.push_back(glm::vec3(size_x, -size_y, size_z)); // 1
    vertices.push_back(glm::vec3(size_x, size_y, size_z)); // 2
    vertices.push_back(glm::vec3(-size_x, size_y, size_z)); // 3

    // back quad
    vertices.push_back(glm::vec3(-size_x, -size_y, -size_z)); // 4
    vertices.push_back(glm::vec3(size_x, -size_y, -size_z)); // 5
    vertices.push_back(glm::vec3(size_x, size_y, -size_z)); // 6
    vertices.push_back(glm::vec3(-size_x, size_y, -size_z)); // 7

#if RENDER_AABBS_WITH_LINES
    // front
    indices.push_back(0U);
    indices.push_back(1U);
    /**/
    indices.push_back(1U);
    indices.push_back(2U);
    /**/
    indices.push_back(2U);
    indices.push_back(3U);
    /**/
    indices.push_back(3U);
    indices.push_back(0U);
    // back
    indices.push_back(4U);
    indices.push_back(5U);
    /**/
    indices.push_back(5U);
    indices.push_back(6U);
    /**/
    indices.push_back(6U);
    indices.push_back(7U);
    /**/
    indices.push_back(7U);
    indices.push_back(4U);
    // side lines
    indices.push_back(1U);
    indices.push_back(5U);
    /**/
    indices.push_back(0U);
    indices.push_back(4U);
    /**/
    indices.push_back(3U);
    indices.push_back(7U);
    /**/
    indices.push_back(2U);
    indices.push_back(6U);
#else // triangles
    // front
    indices.push_back(0U);
    indices.push_back(1U);
    indices.push_back(2U);
    /**/
    indices.push_back(2U);
    indices.push_back(3U);
    indices.push_back(0U);
    // top
    indices.push_back(3U);
    indices.push_back(2U);
    indices.push_back(6U);
    /**/
    indices.push_back(6U);
    indices.push_back(7U);
    indices.push_back(3U);
    // back
    indices.push_back(7U);
    indices.push_back(6U);
    indices.push_back(5U);
    /**/
    indices.push_back(5U);
    indices.push_back(4U);
    indices.push_back(7U);
    // bottom
    indices.push_back(4U);
    indices.push_back(5U);
    indices.push_back(1U);
    /**/
    indices.push_back(1U);
    indices.push_back(0U);
    indices.push_back(4U);
    // left
    indices.push_back(4U);
    indices.push_back(0U);
    indices.push_back(3U);
    /**/
    indices.push_back(3U);
    indices.push_back(7U);
    indices.push_back(4U);
    // right
    indices.push_back(1U);
    indices.push_back(5U);
    indices.push_back(6U);
    /**/
    indices.push_back(6U);
    indices.push_back(2U);
    indices.push_back(1U);
#endif
}

void get_bvh_node_vertex_arrays(const params_t& params,
    std::vector<glm::vec3>& vertices,
    std::vector<int>& indices)
{
    const int numTrianglesInBvh = params.mesh.triangles.size() / 3;
    const int numNodesInBVH = oibvh_get_size(numTrianglesInBvh);
    const int internalNodes = numNodesInBVH - numTrianglesInBvh;

    for (int i = 0; i < numNodesInBVH; ++i) {
        bounding_box_t payload = params.bvh[i];

        // make cube
        std::vector<glm::vec3> cubeVertices;
        std::vector<unsigned int> cubeIndices;

        make_cube(cubeVertices, cubeIndices,
            (payload.maximum.x - payload.minimum.x) / 2.f,
            (payload.maximum.y - payload.minimum.y) / 2.f,
            (payload.maximum.z - payload.minimum.z) / 2.f);

        int backFaceLowerLeftVertexIndex = 4;
        glm::vec3 backFaceLowerLeftVertex = cubeVertices[backFaceLowerLeftVertexIndex];

        glm::vec3 diff = glm::vec3(payload.minimum.x, payload.minimum.y,
                             payload.minimum.z)
            - backFaceLowerLeftVertex; // node->bv.mn - backFaceLowerLeftVertex;

        // shift the bounding box to its real position
        for (int i = 0; i < (int)cubeVertices.size(); ++i) {
            glm::vec3 pos = cubeVertices[i] + diff;
            vertices.push_back(pos);
        }

        // offset indices
        for (int j = 0; j < (int)cubeIndices.size(); ++j) {
            cubeIndices[j] += cubeVertices.size() * i;
        }

        indices.insert(indices.end(), cubeIndices.begin(), cubeIndices.end());
    }
}

void create_bvh_buffer_objects(
    const params_t& params, GLuint& bvhVAO, GLuint& bvhVBO,
    GLuint& bvhIBO,
    std::vector<glm::vec3>& vertices,
    std::vector<int>& indices)
{

    get_bvh_node_vertex_arrays(params, vertices, indices);

    printf("vertices: %d\n", (int)vertices.size());
    printf("indices: %d\n", (int)indices.size());

    glGenVertexArrays(1U, &bvhVAO);
    glGenBuffers(1U, &bvhVBO);
    glGenBuffers(1U, &bvhIBO);

    glBindBuffer(GL_ARRAY_BUFFER, bvhVBO);
    {
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0U);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bvhIBO);
    {
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLint) * indices.size(), indices.data(), GL_STATIC_DRAW);
    }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0U);

    glBindVertexArray(bvhVAO);
    {
        glBindBuffer(GL_ARRAY_BUFFER, bvhVBO);
        glVertexAttribPointer(0U, 3U, GL_FLOAT, GL_FALSE, 0, (void*)0); // position
        glEnableVertexAttribArray(0U);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bvhIBO);
        glEnableVertexAttribArray(0U);
    }
    glBindVertexArray(0U);
}

void process_input(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, delta_time);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, delta_time);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, delta_time);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, delta_time);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        bvhNodeVisualisationRatio += 0.1f * delta_time;
        bvhNodeVisualisationRatio = glm::clamp(bvhNodeVisualisationRatio, 0.f, 1.f);
        printf("bvhNodeVisualisationRatio=%f\n", bvhNodeVisualisationRatio);
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        bvhNodeVisualisationRatio -= 0.1f * delta_time;
        bvhNodeVisualisationRatio = glm::clamp(bvhNodeVisualisationRatio, 0.f, 1.f);
        printf("bvhNodeVisualisationRatio=%f\n", bvhNodeVisualisationRatio);
    }
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
        wireframe = !wireframe;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}

void GLAPIENTRY MessageCallback(GLenum source, GLenum type, GLuint id,
    GLenum severity, GLsizei length,
    const GLchar* message, const void* userParam)
{
    fprintf(stderr,
        "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
        (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""), type, severity,
        message);
}

void glfw_error_CALLBACK(int error, const char* description)
{
    fputs(description, stderr);
}

void oibvh_draw(const params_t& params)
{
    glfwSetErrorCallback(glfw_error_CALLBACK);

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    window = glfwCreateWindow(window_width, window_height, "oibvh draw", NULL, NULL);
    assert(window != NULL);

    glfwMakeContextCurrent(window);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if (monitor) {
        const GLFWvidmode* video_mode = glfwGetVideoMode(monitor);

        // window_width = 384;  //(int)(video_mode->width * 0.5f);
        // window_height = 384; //(int)(video_mode->height * 0.5f);
        glfwSetWindowSize(window, window_width, window_height);
    }

    // load function pointers
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    glfwSwapInterval(0);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    printf("\nrendering info:\n");
    std::pair<const char*, GLenum> info[] = {
        { "OpenGL Version", GL_VERSION },
        { "OpenGL Renderer", GL_RENDERER },
        { "OpenGL GLSL Version", GL_SHADING_LANGUAGE_VERSION },
        { "OpenGL Vendor", GL_VENDOR }
    };
    for (int i(0); i < 4; ++i) {
        printf("\t%s - %s\n", info[i].first, (char*)glGetString(info[i].second));
    }

#ifndef NDEBUG
    glEnable(GL_DEBUG_OUTPUT);
#endif

    glDebugMessageCallback(MessageCallback, 0);
    glViewport(0, 0, window_width, window_height);
    glClearColor(1.f, 1.f, 1.f, 1.f);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LEQUAL);
    glDepthRange(0.0f, 1.0f);
    glFrontFace(GL_CW);

    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //
    // init mesh vertex buffers
    //

    GLint meshShaderProgram;
    GLuint meshVAO;
    GLuint meshVBO;
    GLuint meshIBO;

    meshShaderProgram = graphics_create_program(params.source_files_dir + "/mesh_vertex_shader.glsl", params.source_files_dir + "/mesh_fragment_shader.glsl");

    glGenVertexArrays(1U, &meshVAO);
    glGenBuffers(1U, &meshVBO);
    glGenBuffers(1U, &meshIBO);

    glBindBuffer(GL_ARRAY_BUFFER, meshVBO);
    {
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * params.mesh.vertices.size(), params.mesh.vertices.data(), GL_STATIC_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0U);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshIBO);
    {
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * params.mesh.triangles.size(), params.mesh.triangles.data(), GL_STATIC_DRAW);
    }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0U);

    glBindVertexArray(meshVAO);
    {
        glBindBuffer(GL_ARRAY_BUFFER, meshVBO);
        glVertexAttribPointer(0U, 3U, GL_FLOAT, GL_FALSE, 0, (void*)0); // position

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshIBO);
        glEnableVertexAttribArray(0U);
    }
    glBindVertexArray(0U);

    GLint bvhShaderProgram;
    GLuint bvhVAO;
    GLuint bvhVBO;
    GLuint bvhIBO;

    bvhShaderProgram = graphics_create_program(params.source_files_dir + "/bvh_vertex_shader.glsl",
        params.source_files_dir + "/bvh_fragment_shader.glsl");

    std::vector<int> bvh_mesh_indices;
    std::vector<glm::vec3> bvh_mesh_vertices;
    create_bvh_buffer_objects(params, bvhVAO, bvhVBO, bvhIBO, bvh_mesh_vertices, bvh_mesh_indices);

    do {
        // per-frame time logic
        // --------------------
        float current_time = glfwGetTime();
        delta_time = current_time - prev_time;
        prev_time = current_time;

        // params
        // -----
        process_input(window);

        // render
        // ------

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)window_width / (float)window_height, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 model(1.0);

        GLint location;

        //
        // Draw meshes
        //
        glUseProgram(meshShaderProgram);

        location = glGetUniformLocation(meshShaderProgram, "model");
        glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(model));

        location = glGetUniformLocation(meshShaderProgram, "view");
        glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(view));

        location = glGetUniformLocation(meshShaderProgram, "projection");
        glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(projection));

        location = glGetUniformLocation(meshShaderProgram, "lightColor");
        glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
        glUniform3fv(location, 1, glm::value_ptr(lightColor));

        glm::vec3 lightPos(1.2f, 1.0f, 2.0f);
        location = glGetUniformLocation(meshShaderProgram, "lightPos");
        glUniform3fv(location, 1, /*glm::value_ptr(camera.position)*/ reinterpret_cast<const GLfloat*>(&params.mesh.m_aabb.maximum));

        location = glGetUniformLocation(meshShaderProgram, "viewPos");
        glUniform3fv(location, 1, glm::value_ptr(camera.position));

        glBindVertexArray(meshVAO);
        {
            int wireframe = 0;
            
            location = glGetUniformLocation(meshShaderProgram, "wireframe");
            glUniform1i(location, (wireframe));
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glDrawElements(GL_TRIANGLES, params.mesh.triangles.size(), GL_UNSIGNED_INT, nullptr);
            
            wireframe = 1;
            glUniform1i(location, (wireframe));
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glDrawElements(GL_TRIANGLES, params.mesh.triangles.size(), GL_UNSIGNED_INT, nullptr);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }
        glBindVertexArray(0U);
        glUseProgram(0U);

        //
        // Draw BVH nodes
        //

        glm::mat4 modelViewProjection = projection * view * model;
        glUseProgram(bvhShaderProgram);

        location = glGetUniformLocation(bvhShaderProgram, "modelViewProjection");

        glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(modelViewProjection));

        glBindVertexArray(bvhVAO);
        {
            glDrawElements(
                RENDER_AABBS_WITH_LINES ? GL_LINES : GL_TRIANGLES, 
                bvh_mesh_indices.size() * bvhNodeVisualisationRatio, 
                GL_UNSIGNED_INT, 
                nullptr);
        }
        glBindVertexArray(0U);
        glUseProgram(0U);

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();
    } while (!glfwWindowShouldClose(window));

    //
    // destroy
    //

    if (window) {
        glfwDestroyWindow(window);
    }

    glfwTerminate();
}